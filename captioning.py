import os
import json
import argparse
from torch import nn
import torch.optim as optim
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from PIL import Image
from torchvision import transforms
import gc
import requests
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# Функция для сбора путей изображений и лейблов
def collect_image_paths_and_labels(image_dir):
    folder_paths = []
    labels = []
    for label, folder in enumerate(os.listdir(image_dir)):
        folder_path = os.path.join(image_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for image_name in os.listdir(folder_path):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):
                folder_paths.append(f'{os.path.join(folder, image_name)}')
                labels.append(label)
    return folder_paths, labels

# Функция для генерации подписей и сохранения в JSON
def generate_captions(image_dir, output_file, batch_size=128, prompt=None):
    folder_paths, labels = collect_image_paths_and_labels(image_dir)
    num_images = len(folder_paths)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Инициализация модели
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    
    captions = []

    for i in range(0, num_images, batch_size):
        print(f"Processing batch {i} to {i + batch_size}")
        
        batch_paths = folder_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        images = []

        # Загрузка изображений
        for path in batch_paths:
            try:
                image = Image.open(f"{image_dir}/{path}").convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                images.append(None)

        # Удаление недействительных изображений
        valid_images = [img for img in images if img is not None]
        valid_paths = [path for img, path in zip(images, batch_paths) if img is not None]
        valid_labels = [label for img, label in zip(images, batch_labels) if img is not None]

        # Генерация подписей для изображений
        if valid_images:
            with torch.no_grad():
                if prompt:
                    inputs = processor(valid_images, text=prompt, return_tensors="pt", padding=True).to(device, torch.float16)
                else:
                    inputs = processor(valid_images, return_tensors="pt", padding=True).to(device, torch.float16)

                generated_ids = model.generate(**inputs, max_length=100)
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
            # Добавление путей, подписей и лейблов в список
            for path, text, label in zip(valid_paths, generated_texts, valid_labels):
                captions.append({"image_path": path, "caption": text.strip(), "label": label})

    # Сохранение в JSON-файл
    with open(output_file, 'w') as f:
        json.dump(captions, f, ensure_ascii=False, indent=4)

# Обработка аргументов командной строки
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image captions and save to a JSON file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    
    args = parser.parse_args()
    generate_captions(args.image_dir, args.output_file)
