import os
import gc

import yaml
import random
import secrets
import logging
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import diffusers
from diffusers import (
    AutoencoderKL, EulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline,
)
from diffusers.loaders import AttnProcsLayers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import transformers
import transformers.utils.logging
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from safetensors.torch import save_file

from .model.lora import LoRALinearLayer, LoRACrossAttnProcessor
from .model.utils_sdxl import cast_training_params
from .data.dataset_sdxl import (
    ImageDataset, tokenize_prompt, encode_tokens, compute_time_ids,
)

logger = get_logger(__name__)

BASE_PROMPT = "a photo of {0}"

torch.backends.cuda.enable_flash_sdp(True)


class LoraTrainerSDXL:

    def __init__(self, config):
        self.config = config

    def setup_exp_name(self, exp_idx):
        exp_name = "{0:0>5d}-{1}-{2}".format(
            exp_idx + 1,
            secrets.token_hex(2),
            os.path.basename(os.path.normpath(self.config.train_data_dir)),
        )
        exp_name += f"_lora{self.config.lora_rank}"
        return exp_name

    def setup_exp(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        exp_idx = 0
        for folder in os.listdir(self.config.output_dir):
            try:
                curr_exp_idx = max(exp_idx, int(folder.split("-")[0].lstrip("0")))
                exp_idx = max(exp_idx, curr_exp_idx)
            except Exception:
                pass

        self.config.exp_name = self.setup_exp_name(exp_idx)
        self.config.output_dir = os.path.abspath(
            os.path.join(self.config.output_dir, self.config.exp_name)
        )

        if os.path.exists(self.config.output_dir):
            raise ValueError(
                f"Experiment directory {self.config.output_dir} already exists. Race condition!"
            )
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.config.logging_dir = os.path.join(self.config.output_dir, "logs")
        os.makedirs(self.config.logging_dir, exist_ok=True)

        with open(os.path.join(self.config.logging_dir, "hparams.yml"), "w") as outfile:
            yaml.dump(vars(self.config), outfile)

    def setup_accelerator(self):
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.output_dir
        )
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_project_config,
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def setup_base_model(self):
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler",
            revision=self.config.revision,
            torch_dtype=self.weight_dtype,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.config.revision,
            torch_dtype=self.weight_dtype,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.config.revision,
            torch_dtype=self.weight_dtype,
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=self.config.revision,
            torch_dtype=self.weight_dtype,
        )
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.weight_dtype,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.config.revision,
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=self.config.revision,
        )

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        self.params_to_optimize = []
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
                rank = min(cross_attention_dim, hidden_size, self.config.lora_rank)
            else:
                rank = min(hidden_size, self.config.lora_rank)

            kwargs = {
                "hidden_size": hidden_size,
                "cross_attention_dim": cross_attention_dim,
                "rank": rank,
                "lora_linear_layer": LoRALinearLayer,
            }
            lora_attn_procs[name] = LoRACrossAttnProcessor(**kwargs)

        self.unet.set_attn_processor(lora_attn_procs)
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
        self.accelerator.register_for_checkpointing(self.lora_layers)

        for _, param in self.lora_layers.named_parameters():
            if param.requires_grad:
                self.params_to_optimize.append(param)
        self.lora_layers.train()

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

    def setup_lr_scheduler(self):
        pass

    def setup_dataset(self):
        self.train_dataset = ImageDataset(
            train_data_dir=self.config.train_data_dir,
            resolution=self.config.resolution,
            one_image=self.config.one_image,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=None,
            num_workers=self.config.dataloader_num_workers,
            generator=self.generator,
        )

    def move_to_device(self):
        self.lora_layers, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.lora_layers, self.optimizer, self.train_dataloader
        )
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_2.to(self.accelerator.device, dtype=self.weight_dtype)

        cast_training_params((self.unet), dtype=torch.float32)
        self.unet.enable_gradient_checkpointing()

    def setup_seed(self):
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.config.seed)

    def setup(self):
        self.setup_exp()
        self.setup_accelerator()
        self.setup_seed()
        self.setup_base_model()
        self.setup_model()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.setup_dataset()
        self.move_to_device()
        self.setup_pipeline()

    def prepare_encoder_hidden_states(self):
        self.input_ids_list = tokenize_prompt(
            (self.tokenizer, self.tokenizer_2),
            BASE_PROMPT.format(
                f"{self.config.placeholder_token} {self.config.class_name}"
            ),
        )
        self.encoder_hidden_states, self.pooled_encoder_hidden_states = encode_tokens(
            (self.text_encoder, self.text_encoder_2), self.input_ids_list
        )
        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")
        del self.tokenizer, self.tokenizer_2, self.text_encoder, self.text_encoder_2
        gc.collect()
        torch.cuda.empty_cache()

    def train_step(self, batch):
        latents = self.vae.encode(
            batch["image"].to(self.accelerator.device, dtype=self.weight_dtype) * 2.0 - 1.0
        ).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"].to(self.accelerator.device),
            crops_coords_top_left=batch["crop_top_lefts"].to(self.accelerator.device),
            resolution=self.config.resolution,
        )

        batch_size = latents.shape[0]
        encoder_hidden_states = self.encoder_hidden_states.expand(batch_size, -1, -1)
        pooled_encoder_hidden_states = self.pooled_encoder_hidden_states.expand(batch_size, -1)

        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states,
        }

        outputs = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_conditions,
        ).sample

        loss = F.mse_loss(outputs.float(), target.float(), reduction="mean")
        return loss

    def setup_pipeline(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=scheduler,
            torch_dtype=self.weight_dtype,
        )
        self.pipeline.safety_checker = None
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=False)

    @staticmethod
    def _sanitize_path_component(text):
        bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for ch in bad:
            text = text.replace(ch, '_')
        return text

    def _format_prompt(self, prompt):
        token = f"{self.config.placeholder_token} {self.config.class_name}"
        if '{0}' in prompt:
            return prompt.replace('{0}', token)
        if '{}' in prompt:
            return prompt.replace('{}', token)
        return prompt

    @torch.no_grad()
    def validation(self, epoch):
        generator = torch.Generator(device=self.accelerator.device).manual_seed(42)
        prompts = self.config.validation_prompts.split('#')

        samples_path = os.path.join(
            self.config.output_dir,
            f"checkpoint-{epoch}",
            "samples",
            "ns0_gs0_validation",
            "version_0",
        )
        os.makedirs(samples_path, exist_ok=True)
        self.pipeline = self.pipeline.to(self.accelerator.device)

        for prompt in tqdm(prompts, desc=f"Validation checkpoint-{epoch}"):
            formatted_prompt = self._format_prompt(prompt)
            prompt_dir = os.path.join(samples_path, self._sanitize_path_component(formatted_prompt))
            os.makedirs(prompt_dir, exist_ok=True)
            with torch.autocast("cuda"):
                images = self.pipeline(
                    prompt=formatted_prompt,
                    num_images_per_prompt=self.config.num_val_imgs_per_prompt,
                    guidance_scale=0.0,
                    num_inference_steps=25,
                    generator=generator,
                ).images
            for idx, image in enumerate(images):
                image.save(os.path.join(prompt_dir, f"{idx:02d}.png"))

        torch.cuda.empty_cache()

    def save_model(self, epoch):
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
        os.makedirs(save_path, exist_ok=True)
        state_dict = self.accelerator.get_state_dict(self.lora_layers)
        save_file(state_dict, os.path.join(save_path, "pytorch_lora_weights.safetensors"))

    def train(self):
        self.prepare_encoder_hidden_states()
        for epoch in tqdm(range(self.config.num_train_epochs)):
            batch = next(iter(self.train_dataloader))
            with self.accelerator.autocast():
                loss = self.train_step(batch)

            self.accelerator.backward(loss)

            logs_dict = {"loss": loss.detach().item()}
            self.optimizer.step()
            self.optimizer.zero_grad()

            del batch, loss
            gc.collect()
            torch.cuda.empty_cache()

            for tracker in self.accelerator.trackers:
                tracker.log(logs_dict)

            if self.accelerator.is_main_process:
                if epoch % self.config.checkpointing_steps == 0 and epoch != 0:
                    if self.config.validation_prompts:
                        self.validation(epoch)
                    self.save_model(epoch)

            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        if self.accelerator.is_main_process:
            if self.config.validation_prompts:
                self.validation(self.config.num_train_epochs)
            self.save_model(self.config.num_train_epochs)

        self.accelerator.end_training()
