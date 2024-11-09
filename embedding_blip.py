import os
import json
import argparse
import torch
from transformers import AutoModel
from tqdm import tqdm

def generate_embeddings(input_json, output_json, batch_size=128, task="text-matching"):

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    file_paths = [item['file_path'] for item in data]
    labels = [item['label'] for item in data]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    model.to(device)
    model.eval()
    
    embeddings_list = []
    
    num_texts = len(texts)
    for i in range(0, num_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_file_paths = file_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        
        print(f"Processing batch {i} to {i + len(batch_texts)}")
        
        with torch.no_grad():
            batch_embeddings = model.encode(batch_texts, task=task)
            if isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = batch_embeddings.cpu().numpy()
            else:
                batch_embeddings = [emb.cpu().numpy() for emb in batch_embeddings]
        
        for file_path, embedding, label in zip(batch_file_paths, batch_embeddings, batch_labels):
            embeddings_list.append({
                "file_path": file_path,
                "embedding": embedding.tolist(),  
                "label": label
            })
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(embeddings_list, f, ensure_ascii=False, indent=4)
    print(f"Embeddings saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text embeddings and save to a JSON file.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file containing texts.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing texts.")
    parser.add_argument("--task", type=str, default="text-matching", help="Task type for the model.")
    
    args = parser.parse_args()
    generate_embeddings(args.input_json, args.output_json, batch_size=args.batch_size, task=args.task)