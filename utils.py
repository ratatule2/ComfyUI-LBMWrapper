from typing import List

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from torchvision import transforms

from .lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from .lbm.models.lbm import LBMConfig, LBMModel
from .lbm.models.unets import DiffusersUNet2DCondWrapper
from .lbm.models.vae import AutoencoderKLDiffusers
from diffusers.models import AutoencoderKL

def get_model_from_config(
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_normal",
    selected_timesteps: List[float] = None,
    prob: List[float] = None,
    conditioning_images_keys: List[str] = [],
    conditioning_masks_keys: List[str] = ["mask"],
    source_key: str = "source_image",
    target_key: str = "source_image_paste",
    bridge_noise_sigma: float = 0.0,
):

    conditioners = []

    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,  # Add downsampled_image
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=[1, 2, 10],
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=[5, 10, 20],
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=True,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=None,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch.bfloat16)

    if conditioning_images_keys != [] or conditioning_masks_keys != []:

        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

        # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )

    vae_config = {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.20.0.dev0",
        "_name_or_path": "../sdxl-vae/",
        "act_fn": "silu",
        "block_out_channels": [
            128,
            256,
            512,
            512
        ],
        "down_block_types": [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D"
        ],
        "force_upcast": True,
        "in_channels": 3,
        "latent_channels": 4,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 1024,
        "scaling_factor": 0.13025,
        "up_block_types": [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D"
        ]
        }
    
    vae = AutoencoderKLDiffusers(AutoencoderKL.from_config(vae_config))
    vae.freeze()
    vae.to(torch.bfloat16)

    ## Diffusion Model ##
    # Get diffusion model
    config = LBMConfig(
        source_key=source_key,
        target_key=target_key,
        timestep_sampling=timestep_sampling,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )
    scheduler_config = {
        'num_train_timesteps': 1000, 
        'shift': 1.0, 
        'use_dynamic_shifting': False, 
        'base_shift': 0.5, 'max_shift': 1.15, 
        'base_image_seq_len': 256, 'max_image_seq_len': 4096, 
        'invert_sigmas': False, 'shift_terminal': None, 
        'use_karras_sigmas': False, 'use_exponential_sigmas': False, 
        'use_beta_sigmas': False, 'time_shift_type': 'exponential', 
        '_use_default_values': ['time_shift_type', 'max_image_seq_len', 
        'max_shift', 'base_image_seq_len', 'use_dynamic_shifting', 'invert_sigmas', 
        'shift', 'use_beta_sigmas', 'base_shift', 'use_exponential_sigmas', 
        'shift_terminal'], '_class_name': 'FlowMatchEulerDiscreteScheduler', 
        '_diffusers_version': '0.19.0.dev0', 'beta_end': 0.012, 'beta_schedule': 
        'scaled_linear', 'beta_start': 0.00085, 'clip_sample': False, 'interpolation_type': 
        'linear', 'prediction_type': 'epsilon', 'sample_max_value': 1.0, 'set_alpha_to_one': False, 
        'skip_prk_steps': True, 'steps_offset': 1, 'timestep_spacing': 'leading', 
        'trained_betas': None
    }
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    model = LBMModel(
        config,
        denoiser=denoiser,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)

    return model


def extract_object(birefnet, img):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = img
    input_images = transform_image(image).unsqueeze(0).cuda()

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image = Image.composite(image, Image.new("RGB", image.size, (127, 127, 127)), mask)
    return image, mask


def resize_and_center_crop(image, target_width, target_height):
    original_width, original_height = image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image