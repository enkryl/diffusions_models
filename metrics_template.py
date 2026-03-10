"""Шаблон подсчёта метрик для ДЗ.

Здесь реализована схема из условия:
1) для каждой сгенерированной картинки считаем similarity со всеми тренировочными;
2) score картинки = среднее по тренировочным;
3) score промпта = среднее по картинкам промпта;
4) финальная метрика = среднее по промптам.

ВАЖНО: если на семинаре 1 использовался не CLIP, а другой энкодер, замените функцию
`embed_images` на нужную реализацию — остальная агрегация уже готова.
"""

from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in exts])


def embed_images(paths, model, processor, device="cuda"):
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()


def compute_prompt_scores(train_dir, inference_root, device="cuda"):
    train_paths = list_images(train_dir)
    prompt_dirs = [p for p in Path(inference_root).iterdir() if p.is_dir()]

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    train_emb = embed_images(train_paths, model, processor, device=device)

    prompt_scores = {}
    for prompt_dir in prompt_dirs:
        gen_paths = list_images(prompt_dir)
        gen_emb = embed_images(gen_paths, model, processor, device=device)
        sim = gen_emb @ train_emb.T
        image_scores = sim.mean(dim=1)
        prompt_scores[prompt_dir.name] = {
            "image_scores": image_scores.tolist(),
            "prompt_score": image_scores.mean().item(),
        }

    final_metric = sum(v["prompt_score"] for v in prompt_scores.values()) / max(len(prompt_scores), 1)
    return prompt_scores, final_metric


if __name__ == "__main__":
    train_dir = "my_dataset"
    inference_root = "trained-lora/<experiment>/checkpoint-1000/samples/ns25_gs5.0/version_0"
    prompt_scores, final_metric = compute_prompt_scores(train_dir, inference_root)

    print("Prompt scores:")
    for prompt, info in prompt_scores.items():
        print(f"- {prompt}: {info['prompt_score']:.4f}")
    print(f"Final metric: {final_metric:.4f}")
