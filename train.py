import os
import random
from enum import Enum
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Sampler

from torchvision import transforms
from transformers import AutoModel


def create_image_label_txt(image_dir, output_file):
    with open(output_file, 'w') as file:
        for label, folder in enumerate(os.listdir(image_dir)):
            folder_path = os.path.join(image_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for image_name in os.listdir(folder_path):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    file.write(f"{os.path.join(folder, image_name)};{label}\n")


class ImageLabelDataset(Dataset):
    def __init__(self, image_dir, txt_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_label_list = []

        with open(txt_file, 'r') as file:
            for line in file:
                image_path, label = line.strip().split(';')
                self.image_label_list.append((image_path, int(label)))

    def __len__(self):
        return len(self.image_label_list)

    def __getitem__(self, idx):
        image_path, label = self.image_label_list[idx]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class MPerClassSampler(Sampler):
    def __init__(self, dataset, m, batch_size):
        self._m_per_class = m
        self._batch_size = batch_size
        self._labels_to_indices = self._get_labels_to_indices(dataset)
        self._global_labels = list(self._labels_to_indices.keys())
        self.labels = self._global_labels

        assert (self._batch_size % self._m_per_class) == 0, "m_per_class must divide batch_size without remainder"
        self._sample_length = self._get_sample_length()

    def __iter__(self):
        idx_list = [0] * self._sample_length
        i = 0
        num_iters = self.num_iters()
        for _ in range(num_iters):
            random.shuffle(self.labels)
            curr_label_set = self.labels[: self._batch_size // self._m_per_class]
            for label in curr_label_set:
                t = self._labels_to_indices[label].copy()
                random.shuffle(t)
                idx_list[i : i + self._m_per_class] = t[: self._m_per_class]
                i += self._m_per_class
        return iter(idx_list)

    def num_iters(self):
        return self._sample_length // self._batch_size

    def _get_sample_length(self):
        sample_length = sum([len(self._labels_to_indices[k]) for k in self.labels])
        sample_length -= sample_length % self._batch_size
        return sample_length

    def _get_labels_to_indices(self, dataset):
        labels_to_indices = {}
        for index, (_, label) in enumerate(dataset):
            if label not in labels_to_indices:
                labels_to_indices[label] = []
            labels_to_indices[label].append(index)
        return labels_to_indices

    def __len__(self):
        return self._sample_length


class ImageProjectionModel(nn.Module):
    def __init__(self, base_model, input_dim=768, projection_dim=128):
        super(ImageProjectionModel, self).__init__()
        self.base_model = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )

    def forward(self, images):
        # processed_images = [toPIL(img) for img in images]
        with torch.no_grad():
            embeddings = self.base_model.get_image_features(images)
        projections = self.projection_head(embeddings)
        return projections


from torchvision import models

class EfficientNetEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(EfficientNetEmbedding, self).__init__()

        self.efficient_net = models.efficientnet_b0(pretrained=True)

        num_features = self.efficient_net.classifier[1].in_features
        self.efficient_net.classifier = nn.Sequential(
            nn.Linear(num_features, embed_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.efficient_net(x)




class Difficulty(Enum):
    Easy = 1        # A - P < A - N
    SemiHard = 2    # min(A - N)
    Hard = 3        # max(A - P), min(A - N)


def _get_anchor_positive_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


def _pairwise_distances(embeddings, squared=False, cosine=False):
    dot_product = torch.matmul(embeddings, embeddings.t())
    if cosine: # Cosine range is -1 to 1. 1 - similarity makes 0 be closest, 2 = furthest
        norm = torch.norm(embeddings, dim=1, keepdim=True)
        similarity = dot_product / torch.matmul(norm, norm.t())
        return 1 - similarity

    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances[distances < 0] = 0
    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16
        distances = (1.0 - mask) * torch.sqrt(distances)
    return distances


def _masked_minimum(data, mask, dim=1):
    axis_maximums = data.max(dim, keepdim=True).values
    masked_minimums = ((data - axis_maximums) * mask.float()).min(dim, keepdim=True).values + axis_maximums
    return masked_minimums


def _masked_maximum(data, mask, dim=1):
    axis_minimums = data.min(dim, keepdim=True).values
    masked_maximums = ((data - axis_minimums) * mask.float()).max(dim, keepdim=True).values + axis_minimums
    return masked_maximums


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False, cosine=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared, cosine=cosine)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()
    return triplet_loss


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2, cosine=False, difficulty=Difficulty.Easy):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cosine = cosine
        self.difficulty = difficulty

    # e.g. loss.change_parameter(difficulty=Difficulty.Hard)
    def change_parameter(self, margin=None, cosine=None, difficulty=None):
        self.margin = self.margin if margin is None else margin
        self.cosine = self.cosine if cosine is None else cosine
        self.difficulty = self.difficulty if difficulty is None else difficulty

    def forward(self, labels, embeddings):
        if self.difficulty == Difficulty.Hard:
            return batch_hard_triplet_loss(labels, embeddings, self.margin, cosine=self.cosine)

        adjacency_not = _get_anchor_negative_triplet_mask(labels)
        batch_size = labels.size(0)

        pdist_matrix = _pairwise_distances(embeddings, cosine=self.cosine)
        pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
        mask = adjacency_not.repeat(batch_size, 1)

        if self.difficulty == Difficulty.Easy:
            mask = mask & torch.gt(pdist_matrix_tile, pdist_matrix.t().reshape(-1, 1))

        mask_final = torch.gt(mask.float().sum(dim=1, keepdim=True), 0.0).reshape(batch_size, batch_size)
        mask_final = mask_final.t()

        adjacency_not = adjacency_not.float()
        mask = mask.float()

        negatives_outside = (
            _masked_minimum(pdist_matrix_tile, mask)
            .reshape(batch_size, batch_size)
            .t()
        )

        negatives_inside = _masked_maximum(pdist_matrix, adjacency_not).repeat(1, batch_size)
        semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = self.margin + pdist_matrix - semi_hard_negatives

        mask_positives = _get_anchor_positive_triplet_mask(labels)
        num_positives = torch.sum(mask_positives)
        triplet_loss = torch.sum(torch.clamp(loss_mat * mask_positives, min=0.0)) / (num_positives + 1e-8)
        return triplet_loss


def train(model, optimizer, criterion, scheduler, train_loader, test_loader, num_epochs, device, epochs_dir):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        train_embeddings = []
        train_labels = []
        for imgs, labels in train_loader_tqdm:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            embeddings = model(imgs)

            loss = criterion(labels, embeddings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_embeddings.append(embeddings.cpu())
            train_labels.append(labels.cpu())

            train_loader_tqdm.set_postfix(loss=loss.item())

        train_embeddings = torch.cat(train_embeddings)
        train_labels = torch.cat(train_labels)

        scheduler.step()
        model.eval()
        val_loss = 0


        val_embeddings = []
        val_labels = []
        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")

        with torch.no_grad():
            for imgs, labels in test_loader_tqdm:
                imgs, labels = imgs.to(device), labels.to(device)

                embeddings = model(imgs)

                loss = criterion(labels, embeddings)
                val_loss += loss.item()

                val_embeddings.append(embeddings.cpu())
                val_labels.append(labels.cpu())

                test_loader_tqdm.set_postfix(loss=loss.item())

        val_embeddings = torch.cat(val_embeddings)
        val_labels = torch.cat(val_labels)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(test_loader)}")
        torch.save(model.state_dict(), f"{epochs_dir}/triplet_weights_epoch_{epoch}.pth")

        matching_ratios = []
        average_precisions = []

        for i in range(len(val_embeddings)):
            val_emb = val_embeddings[i]
            val_label = val_labels[i]

            distances = torch.nn.functional.cosine_similarity(train_embeddings, val_emb.unsqueeze(0), dim=1)
            nearest_indices = torch.topk(distances, k=10, largest=False).indices
            nearest_labels = train_labels[nearest_indices]

            num_matching = (nearest_labels == val_label).sum().item()
            matching_ratio = num_matching / 10.0
            matching_ratios.append(matching_ratio)

            relevant_indices = (nearest_labels == val_label).nonzero(as_tuple=True)[0]
            precisions = [(rank + 1) / (index + 1) for rank, index in enumerate(relevant_indices.tolist())]
            average_precision = sum(precisions) / len(precisions) if precisions else 0
            average_precisions.append(average_precision)

        average_matching_ratio = sum(matching_ratios) / len(matching_ratios)
        map10 = sum(average_precisions) / len(average_precisions)
        print("\nValidation metric based on nearest neighbors:")
        print(f"Average matching ratio: {average_matching_ratio:.4f}")
        print(f"MAP@10: {map10:.4f}")



def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training script for a neural network")

    parser.add_argument("--image_dir", type=str, default="../sekrrno/dataset", help="image dir")
    parser.add_argument("--epochs_dir", type=str, default="./epochs", help="epochs dir")

    args = parser.parse_args()

    image_dir = args.image_dir
    epochs_dir = args.epochs_dir
    create_directory_if_not_exists(epochs_dir)
    output_file = "image_labels.txt"
    create_image_label_txt(image_dir, output_file)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])

    dataset = ImageLabelDataset(image_dir=image_dir, txt_file=output_file, transform=transform)


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    base_model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)

    for param in base_model.parameters():
        param.requires_grad = False

    

    n_blocks_to_unfreeze = 2 

    blocks = base_model.vision_model.blocks

    for block in blocks[-n_blocks_to_unfreeze:]:
        for param in block.parameters():
            param.requires_grad = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")


    m_per_class = 4
    batch_size = 128
    sampler_train = MPerClassSampler(dataset=train_dataset, m=m_per_class, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    model = ImageProjectionModel(base_model, projection_dim=64).to(device)
    base_model_params = [param for param in base_model.parameters() if param.requires_grad]
    projection_head_params = model.projection_head.parameters()

    optimizer = optim.Adam([
        {'params': projection_head_params, 'lr': 1e-4},
        {'params': base_model_params, 'lr': 1e-5}  
    ])
    criterion = TripletLoss(margin=0.5, difficulty=Difficulty.Hard, cosine=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    model.to(device)

    num_epochs = 4
    train(model, 
          optimizer, 
          criterion, 
          scheduler, 
          train_loader, 
          test_loader, 
          num_epochs, 
          device, 
          epochs_dir)


    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # model.eval()

    # def compute_embeddings(dataloader):
    #     embeddings = []
    #     labels = []
    #     with torch.no_grad():
    #         for imgs, lbls in tqdm(dataloader, desc="Computing embeddings"):
    #             imgs = imgs.to(device)
    #             emb = model(imgs)
    #             embeddings.append(emb.cpu())
    #             labels.append(lbls)
    #     embeddings = torch.cat(embeddings)
    #     labels = torch.cat(labels)
    #     return embeddings, labels

    # train_embeddings, train_labels = compute_embeddings(train_loader)
    # val_embeddings, val_labels = compute_embeddings(test_loader)

    # matching_ratios = []
    # average_precisions = []
    # for i in tqdm(range(len(val_embeddings)), desc="Evaluating validation samples"):
    #     val_emb = val_embeddings[i]
    #     val_label = val_labels[i]

    #     distances = torch.nn.functional.cosine_similarity(train_embeddings, val_emb.unsqueeze(0), dim=1)

    #     nearest_indices = torch.topk(distances, k=10, largest=False).indices

    #     nearest_labels = train_labels[nearest_indices]

    #     num_matching = (nearest_labels == val_label).sum().item()

    #     matching_ratio = num_matching / 10.0

    #     matching_ratios.append(matching_ratio)
    #     relevant_indices = (nearest_labels == val_label).nonzero(as_tuple=True)[0]  # Positions where labels match
    #     precisions = [(rank + 1) / (index + 1) for rank, index in enumerate(relevant_indices.tolist())]
    #     average_precision = sum(precisions) / len(precisions) if precisions else 0
    #     average_precisions.append(average_precision)

    # average_matching_ratio = sum(matching_ratios) / len(matching_ratios)
    # map10 = sum(average_precisions) / len(average_precisions)
    # print("\nValidation metric based on nearest neighbors:")
    # print(f"Average matching ratio: {average_matching_ratio:.4f}")
    # print(f"MAP@10: {map10:.4f}")