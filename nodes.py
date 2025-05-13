import os
import torch
from tqdm import tqdm

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

from .utils import get_model_from_config

script_directory = os.path.dirname(os.path.abspath(__file__))


class LoadLBMModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "load_device": (["main_device", "offload_device"], {"default": "cuda", "tooltip": "Initialize the model on the main device or offload device"}),
            },
        }
        

    RETURN_TYPES = ("LBM_MODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "LBMWrapper"

    def loadmodel(self, model, base_precision, load_device="main_device"):

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        if load_device == "main_device":
            transformer_load_device = device
        else:
            transformer_load_device = offload_device

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
      
        config = {
            "vae_num_channels": 4,
            "unet_input_channels": 4,
            "timestep_sampling": "custom_timesteps",
            "selected_timesteps": [250, 500, 750, 1000],
            "prob": [0.25, 0.25, 0.25, 0.25],
            "conditioning_images_keys": [],
            "conditioning_masks_keys": [],
            "source_key": "source_image",
            "target_key": "source_image_paste",
            "bridge_noise_sigma": 0.005,
        }
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)
        
        with init_empty_weights():
            unet = get_model_from_config(**config)

        print("Using accelerate to load and assign model weights to device...")
        param_count = sum(1 for _ in unet.named_parameters())
        for name, param in tqdm(unet.named_parameters(),
                desc=f"Loading transformer parameters to {transformer_load_device}",
                total=param_count,
                leave=True):
            set_module_tensor_to_device(unet, name, device=transformer_load_device, dtype=base_dtype, value=sd[name])

        return(unet, )


class LBMSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("LBM_MODEL",),
                "image": ("IMAGE", ),
                "steps": ("INT", {"default": 30, "min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "LBMWrapper"

    def process(self, model, image, steps):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        input_image = image.clone().permute(0, 3, 1, 2).to(device, model.dtype) * 2 - 1

        batch = {
            "source_image": input_image,
        }
        model.vae.to(device)
        z_source = model.vae.encode(batch[model.source_key])
        model.vae.to(offload_device)

        model.to(device)

        result = model.sample(
            z=z_source,
            num_steps=steps,
            conditioner_inputs=batch,
            max_samples=1,
        ).clamp(-1, 1)

        out = result.permute(0, 2, 3, 1).cpu().float()
        out = (out + 1) / 2

        model.to(offload_device)
        mm.soft_empty_cache()

        return out,

NODE_CLASS_MAPPINGS = {
    "LoadLBMModel": LoadLBMModel,
    "LBMSampler": LBMSampler,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLBMModel": "Load LBM Model",
    "LBMSampler": "LBMSampler",
    }

