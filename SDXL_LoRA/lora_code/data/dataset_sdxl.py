import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Resize, InterpolationMode, ToTensor, RandomCrop, RandomHorizontalFlip
)


def tokenize_prompt(tokenizers, prompt):
    text_input_ids_list = []
    for tokenizer in tokenizers:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_list.append(text_inputs.input_ids)
    return text_input_ids_list


def encode_tokens(text_encoders, text_input_ids_list):
    prompt_embeds_list = []

    for text_encoder, text_input_ids in zip(text_encoders, text_input_ids_list):
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True, return_dict=False
        )

        # Note: We are only ALWAYS interested in the pooled output of the final text encoder
        # (batch_size, pooled_dim)
        pooled_prompt_embeds = prompt_embeds[0]
        # (batch_size, seq_len, dim)
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    # (batch_size, seq_len, dim)
    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    # (batch_size, pooled_dim)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def compute_time_ids(original_size, crops_coords_top_left, resolution):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = torch.tensor([[resolution, resolution]], device=original_size.device)
    target_size = target_size.expand_as(original_size)

    add_time_ids = torch.cat([original_size, crops_coords_top_left, target_size], dim=1)
    return add_time_ids


class ImageDataset(Dataset):
    def __init__(
        self,
        train_data_dir,
        resolution=1024,
        rand=False,
        repeats=100,
        one_image=False,
    ):
        self.train_data_dir = train_data_dir
        self.data_fnames = [
            os.path.join(r, f) for r, d, fs in os.walk(self.train_data_dir)
            for f in fs
        ]
        self.data_fnames = sorted(self.data_fnames)
        if one_image:
            self.data_fnames = [a for a in self.data_fnames if a.endswith(one_image)]

        self.num_images = len(self.data_fnames)
        self._length = self.num_images * repeats

        self.resolution = resolution
        self.rand = rand

    def process_img(self, img):
        image = Image.open(img).convert('RGB')
        w, h = image.size
        crop = min(w, h)
        if self.rand:
            image = Resize(560, interpolation=InterpolationMode.BILINEAR, antialias=True)(image)
            image = RandomCrop(self.resolution)(image)
            image = RandomHorizontalFlip()(image)
        else:
            image = image.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        input_img = torch.cat([ToTensor()(image)])

        return input_img, torch.tensor([crop, crop])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        image_file = self.data_fnames[index % self.num_images]
        input_img, example["original_sizes"] = self.process_img(image_file)
        # assert example["original_sizes"][0] == example["original_sizes"][1], \
        #     'SDXL has a complicated procedure to handle rectangle images. We do not implement it'
        example["crop_top_lefts"] = torch.tensor([0, 0])

        example['image_path'] = image_file
        example['image'] = input_img

        return example
