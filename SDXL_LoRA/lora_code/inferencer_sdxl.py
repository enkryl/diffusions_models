import os
from tqdm import tqdm
import random
import numpy as np
import gc

import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
)
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from safetensors.torch import load_file

from .model.lora import (
    LoRALinearLayer,
    LoRACrossAttnProcessor,
)
from .utils.seed import get_seed


class LoraInferencerSDXL:
    def __init__(self, config, args, prompts, dtype=torch.bfloat16, device="cuda"):
        self.config = config
        self.args = args
        self.checkpoint_idx = args.checkpoint_idx
        self.num_images_per_prompt = args.num_images_per_prompt
        self.batch_size = args.batch_size

        if self.checkpoint_idx is None:
            self.checkpoint_path = config["output_dir"]
        else:
            self.checkpoint_path = os.path.join(
                config["output_dir"], f"checkpoint-{self.checkpoint_idx}"
            )

        self.prompts = prompts
        self.replace_inference_output = self.args.replace_inference_output
        self.version = self.args.version
        self.device = device
        self.dtype = dtype

    def setup_pipe_kwargs(self):
        self.pipe_kwargs = {
            "guidance_scale": self.args.guidance_scale,
            "num_inference_steps": self.args.num_inference_steps,
        }

    def setup_base_model(self):
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            torch_dtype=self.dtype,
            subfolder="scheduler",
            revision=self.config["revision"],
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            subfolder="unet",
            revision=self.config["revision"],
            torch_dtype=self.dtype,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            subfolder="text_encoder",
            revision=self.config["revision"],
            torch_dtype=self.dtype,
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            subfolder="text_encoder_2",
            revision=self.config["revision"],
            torch_dtype=self.dtype,
        )
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.dtype,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            subfolder="tokenizer",
            revision=self.config["revision"],
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            subfolder="tokenizer_2",
            revision=self.config["revision"],
        )

    @staticmethod
    def _sanitize_path_component(text):
        bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for ch in bad:
            text = text.replace(ch, '_')
        return text

    def _format_prompt(self, prompt):
        token = f"{self.config['placeholder_token']} {self.config['class_name']}"
        if '{0}' in prompt:
            return prompt.replace('{0}', token)
        if '{}' in prompt:
            return prompt.replace('{}', token)
        return prompt

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            else:
                raise ValueError(f"Unexpected attention processor name: {name}")

            if cross_attention_dim:
                rank = min(cross_attention_dim, hidden_size, self.config["lora_rank"])
            else:
                rank = min(hidden_size, self.config["lora_rank"])

            kwargs = {
                "hidden_size": hidden_size,
                "cross_attention_dim": cross_attention_dim,
                "rank": rank,
                "lora_linear_layer": LoRALinearLayer,
            }
            lora_attn_procs[name] = LoRACrossAttnProcessor(**kwargs)

        self.unet.set_attn_processor(lora_attn_procs)
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)

        lora_path = os.path.join(self.checkpoint_path, "pytorch_lora_weights.safetensors")
        state_dict = load_file(lora_path)
        self.lora_layers.load_state_dict(state_dict, strict=True)

    def setup_pipeline(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            revision=self.config.get("revision"),
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.scheduler,
            torch_dtype=self.dtype,
        )
        self.pipe.to(self.device)
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=False)

    def setup(self):
        self.setup_base_model()
        self.setup_model()
        self.setup_pipeline()
        self.setup_pipe_kwargs()
        self.create_folder_name()
        self.setup_paths()

    def create_folder_name(self):
        self.inference_folder_name = (
            f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}"
        )

    def setup_paths(self):
        if self.version is None:
            version = 0
            samples_path = os.path.join(
                self.checkpoint_path,
                "samples",
                self.inference_folder_name,
                f"version_{version}",
            )
            while os.path.exists(samples_path):
                version += 1
                samples_path = os.path.join(
                    self.checkpoint_path,
                    "samples",
                    self.inference_folder_name,
                    f"version_{version}",
                )
        else:
            samples_path = os.path.join(
                self.checkpoint_path,
                "samples",
                self.inference_folder_name,
                f"version_{self.version}",
            )
        self.samples_path = samples_path

    def check_generation(self, path, num_images_per_prompt):
        if self.replace_inference_output:
            return True
        if os.path.exists(path) and len(os.listdir(path)) == num_images_per_prompt:
            return False
        return True

    @torch.no_grad()
    def generate_with_prompt(self, prompt, num_images_per_prompt, batch_size):
        n_batches = (num_images_per_prompt - 1) // batch_size + 1
        images = []
        formatted_prompt = self._format_prompt(prompt)

        for i in range(n_batches):
            seed = get_seed(formatted_prompt, i, self.args.seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            generator = torch.Generator(device=self.device).manual_seed(seed)
            current_batch = min(batch_size, num_images_per_prompt - len(images))
            with torch.autocast("cuda"):
                batch_images = self.pipe(
                    prompt=formatted_prompt,
                    num_images_per_prompt=current_batch,
                    generator=generator,
                    **self.pipe_kwargs,
                ).images
            images.extend(batch_images[:current_batch])

            gc.collect()
            torch.cuda.empty_cache()
        return images

    def save_images(self, images, path):
        os.makedirs(path, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(path, f"{idx:02d}.png"))

    def generate_with_prompt_list(self, prompts, num_images_per_prompt, batch_size):
        for prompt in tqdm(prompts, desc="Inference prompts"):
            formatted_prompt = self._format_prompt(prompt)
            path = os.path.join(self.samples_path, self._sanitize_path_component(formatted_prompt))
            if self.check_generation(path, num_images_per_prompt):
                images = self.generate_with_prompt(
                    prompt, num_images_per_prompt, batch_size
                )
                self.save_images(images, path)

    def generate(self):
        self.generate_with_prompt_list(
            self.prompts, self.num_images_per_prompt, self.batch_size
        )
