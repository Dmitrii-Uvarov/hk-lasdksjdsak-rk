import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, ViTImageProcessor, CLIPProcessor, CLIPModel
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import List

# Your existing classes and methods (ImageLabelDataset, Embedder, etc.) remain the same

# Function to compute embeddings


def compute_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            img_feat = models['trunk'](images)
            emb = models['embedder'](img_feat)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

# Function to perform k-NN search


def get_top_k_neighbors(embeddings, k=50):
    knn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(embeddings)
    distances, indices = knn.kneighbors(embeddings)
    return indices

# Function to rerank using CLIP


def rerank_with_clip(query_image_path, candidate_image_paths, clip_model, clip_processor, device):
    query_image = clip_processor(images=Image.open(query_image_path), return_tensors="pt").to(device)
    candidate_images = [
        clip_processor(images=Image.open(img_path), return_tensors="pt")['pixel_values'].to(device)
        for img_path in candidate_image_paths
    ]

    with torch.no_grad():
        query_features = clip_model.get_image_features(**query_image)
        candidate_features = torch.stack([
            clip_model.get_image_features(pixel_values=img).squeeze(0)
            for img in candidate_images
        ])

    similarities = torch.cosine_similarity(query_features, candidate_features)
    top10_indices = similarities.argsort(descending=True)[:10]
    return [candidate_image_paths[i] for i in top10_indices]

# Function to calculate MAP@10


def mean_average_precision_at_k(actual: List[int], predicted: List[List[int]], k=10) -> float:
    avg_precision = 0
    for i in range(len(actual)):
        relevant_items = 0
        precision_sum = 0
        for j in range(k):
            if predicted[i][j] == actual[i]:
                relevant_items += 1
                precision_sum += relevant_items / (j + 1)
        if relevant_items > 0:
            avg_precision += precision_sum / relevant_items
    return avg_precision / len(actual)


class ImageLabelDataset(Dataset):
    def __init__(self, image_preprocessor, image_dir, txt_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_label_list = []
        self.image_preprocessor = image_preprocessor

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

        image = self.image_preprocessor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)

        return image, label


class Embedder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Embedder, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        return self.projection_head(x.last_hidden_state)


def load_last_checkpoint(models, checkpoint_path):
    """Load the last checkpoint if available."""
    checkpoint = torch.load(checkpoint_path)
    models['trunk'].load_state_dict(checkpoint['trunk_state_dict'])
    models['embedder'].load_state_dict(checkpoint['embedder_state_dict'])
    return checkpoint['epoch'] + 1


if __name__ == "__main__":
    image_dir = "../sekrrno/dataset"
    output_file = "image_labels.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trunk = ViTModel.from_pretrained('facebook/dino-vits16')
    trunk_output_size = trunk.config.hidden_size

    trunk = nn.DataParallel(trunk).to(device)
    image_processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')

    dataset = ImageLabelDataset(image_preprocessor=image_processor, image_dir=image_dir,
                                txt_file=output_file)  # , transform=transform)

    embedder = nn.DataParallel(Embedder(input_dim=trunk_output_size, embedding_dim=64)).to(device)

    models = {'trunk': trunk, 'embedder': embedder}

    load_last_checkpoint(models, '~/environments/hack/hk-lasdksjdsak-rk/checkpoint_epoch_26.pth')

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_labels = [dataset.image_label_list[idx][1] for idx in train_dataset.indices]
    test_labels = [dataset.image_label_list[idx][1] for idx in test_dataset.indices]

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    train_embeddings = compute_embeddings(models, train_loader, device)
    test_embeddings = compute_embeddings(models, test_loader, device)

    top50_indices = get_top_k_neighbors(train_embeddings, k=50)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    reranked_top10 = []
    for i, (query_img_path, label) in enumerate(test_dataset.image_label_list):
        candidate_paths = [train_dataset.image_label_list[idx][0] for idx in top50_indices[i]]
        top10_images = rerank_with_clip(query_img_path, candidate_paths, clip_model, clip_processor, device)
        reranked_top10.append([train_dataset.image_label_list.index(img) for img in top10_images])

    map_reranked = mean_average_precision_at_k(test_labels, reranked_top10)

    original_top10 = [indices[:10] for indices in top50_indices]
    map_original = mean_average_precision_at_k(test_labels, original_top10)

    print(f"MAP@10 with CLIP reranking: {map_reranked}")
    print(f"MAP@10 without CLIP reranking: {map_original}")