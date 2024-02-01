import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union, Dict, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import (DPMSolverMultistepScheduler,
                       StableDiffusionControlNetInpaintPipeline,
                       StableDiffusionXLPipeline)
from easyphoto.easyphoto_config import preload_lora
from easyphoto.train_kohya.utils.model_utils import \
    load_models_from_stable_diffusion_checkpoint
from safetensors.torch import load_file
from transformers import CLIPTokenizer
from processing import StableDiffusionProcessingImg2Img

tokenizer       = None
scheduler       = None
text_encoder    = None
vae             = None
unet            = None
pipeline        = None
sd_model_checkpoint_before  = ""
weight_dtype                = torch.float16
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


def merge_lora(pipeline, lora_path, multiplier, from_safetensor=False, device='cpu', dtype=torch.float32):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if from_safetensor:
        state_dict = load_file(lora_path, device=device)
    else:
        checkpoint = torch.load(os.path.join(lora_path, 'pytorch_lora_weights.bin'), map_location=torch.device(device))
        new_dict = dict()
        for idx, key in enumerate(checkpoint):
            new_key = re.sub(r'\.processor\.', '_', key)
            new_key = re.sub(r'mid_block\.', 'mid_block_', new_key)
            new_key = re.sub('_lora.up.', '.lora_up.', new_key)
            new_key = re.sub('_lora.down.', '.lora_down.', new_key)
            new_key = re.sub(r'\.(\d+)\.', '_\\1_', new_key)
            new_key = re.sub('to_out', 'to_out_0', new_key)
            new_key = 'lora_unet_' + new_key
            new_dict[new_key] = checkpoint[key]
            state_dict = new_dict
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(layer_infos) == 0:
                    print('Error loading layer')
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        curr_layer.weight.data = curr_layer.weight.data.to(device)
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2),
                                                                    weight_down.squeeze(3).squeeze(2)).unsqueeze(
                2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline


# TODO: Refactor with merge_lora.
def unmerge_lora(pipeline, lora_path, multiplier=1, from_safetensor=False, device="cpu", dtype=torch.float32):
    """Unmerge state_dict in LoRANetwork from the pipeline in diffusers."""
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if from_safetensor:
        state_dict = load_file(lora_path, device=device)
    else:
        checkpoint = torch.load(os.path.join(lora_path, 'pytorch_lora_weights.bin'), map_location=torch.device(device))
        new_dict = dict()
        for idx, key in enumerate(checkpoint):
            new_key = re.sub(r'\.processor\.', '_', key)
            new_key = re.sub(r'mid_block\.', 'mid_block_', new_key)
            new_key = re.sub('_lora.up.', '.lora_up.', new_key)
            new_key = re.sub('_lora.down.', '.lora_down.', new_key)
            new_key = re.sub(r'\.(\d+)\.', '_\\1_', new_key)
            new_key = re.sub('to_out', 'to_out_0', new_key)
            new_key = 'lora_unet_' + new_key
            new_dict[new_key] = checkpoint[key]
            state_dict = new_dict

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(layer_infos) == 0:
                    print('Error loading layer')
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        curr_layer.weight.data = curr_layer.weight.data.to(device)
        if len(weight_up.shape) == 4:
            curr_layer.weight.data -= multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2),
                                                                    weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data -= multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline


def t2i_sdxl_call(
    steps=20,
    seed=-1,

    cfg_scale=7.0,
    width=640,
    height=768,

    prompt="",
    negative_prompt="",
    sd_model_checkpoint="",
):  
    width   = int(width // 8 * 8)
    height  = int(height // 8 * 8)

    # Load scheduler, tokenizer and models.
    sdxl_pipeline = StableDiffusionXLPipeline.from_single_file(sd_model_checkpoint).to("cuda", weight_dtype)
    sdxl_pipeline.scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012)

    try:
        import xformers
        sdxl_pipeline.enable_xformers_memory_efficient_attention()
    except:
        logging.warning('No module named xformers. Infer without using xformers. You can run pip install xformers to install it.')

    generator = torch.Generator("cuda").manual_seed(int(seed)) 

    image = sdxl_pipeline(
        prompt, negative_prompt=negative_prompt, 
        guidance_scale=cfg_scale, num_inference_steps=steps, generator=generator, height=height, width=width
    ).images[0]

    del sdxl_pipeline
    torch.cuda.empty_cache()
    return image


InputImage = Union[np.ndarray, str]
InputImage = Union[Dict[str, InputImage], Tuple[InputImage, InputImage], InputImage]


class ControlNetUnit:
    """
    Represents an entire ControlNet processing unit.
    """

    def __init__(
        self,
        enabled: bool = True,
        module: Optional[str] = None,
        model: Optional[str] = None,
        weight: float = 1.0,
        image: Optional[InputImage] = None,
        resize_mode: Union[int, str] = 1,
        low_vram: bool = False,
        processor_res: int = -1,
        threshold_a: float = -1,
        threshold_b: float = -1,
        guidance_start: float = 0.0,
        guidance_end: float = 1.0,
        pixel_perfect: bool = False,
        control_mode: Union[int, str] = 0,
        save_detected_map: bool = True,
        batch_images=[],
        **_kwargs,
    ):
        self.enabled = enabled
        self.module = module
        self.model = model
        self.weight = weight
        self.image = image
        self.resize_mode = resize_mode
        self.low_vram = low_vram
        self.processor_res = processor_res
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.pixel_perfect = pixel_perfect
        self.control_mode = control_mode
        self.save_detected_map = save_detected_map
        self.batch_images = batch_images

    def __eq__(self, other):
        if not isinstance(other, ControlNetUnit):
            return False

        return vars(self) == vars(other)


def i2i_inpaint_call(
        images=[],
        mask_image=None,
        denoising_strength=0.75,
        # controlnet_image=[],
        # controlnet_units_list=[],
        # controlnet_conditioning_scale=[],
        steps=20,
        seed=-1,
        cfg_scale=7.0,
        width=640,
        height=768,
        prompt="",
        negative_prompt="",
        # sd_lora_checkpoint=[],
        sd_model_checkpoint="Chilloutmix-Ni-pruned-fp16-fix.safetensors",
        # sd_base15_checkpoint="",

        resize_mode=0,
        image_cfg_scale=1.5,
        mask_blur=8,
        inpainting_fill=0,
        inpaint_full_res=True,
        inpaint_full_res_padding=0,
        inpainting_mask_invert=0,
        initial_noise_multiplier=1,
        styles=[],
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        batch_size=1,
        n_iter=1,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=False,
        eta=1.0,
        s_churn=0,
        s_tmax=0,
        s_tmin=0,
        s_noise=1,
        override_settings={},
        override_settings_restore_afterwards=True,
        sampler=None,
        include_init_images=False,
        controlnet_units: List[ControlNetUnit] = [],
        use_deprecated_controlnet=False,
        outpath_samples="",
        sd_vae="vae-ft-mse-840000-ema-pruned.ckpt",
):
    """
        Perform image-to-image inpainting.

        Args:
            images (list): List of input images.
            resize_mode (int): Resize mode.
            denoising_strength (float): Denoising strength.
            image_cfg_scale (float): Image configuration scale.
            mask_image (PIL.Image.Image): Mask image.
            mask_blur (int): Mask blur strength.
            inpainting_fill (int): Inpainting fill value.
            inpaint_full_res (bool): Flag to inpaint at full resolution.
            inpaint_full_res_padding (int): Padding size for full resolution inpainting.
            inpainting_mask_invert (int): Invert the mask flag.
            initial_noise_multiplier (int): Initial noise multiplier.
            prompt (str): Prompt text.
            styles (list): List of styles.
            seed (int): Seed value.
            subseed (int): Subseed value.
            subseed_strength (int): Subseed strength.
            seed_resize_from_h (int): Seed resize height.
            seed_resize_from_w (int): Seed resize width.
            batch_size (int): Batch size.
            n_iter (int): Number of iterations.
            steps (list): List of steps.
            cfg_scale (float): Configuration scale.
            width (int): Output image width.
            height (int): Output image height.
            restore_faces (bool): Restore faces flag.
            tiling (bool): Tiling flag.
            do_not_save_samples (bool): Do not save samples flag.
            do_not_save_grid (bool): Do not save grid flag.
            negative_prompt (str): Negative prompt text.
            eta (float): Eta value.
            s_churn (int): Churn value.
            s_tmax (int): Tmax value.
            s_tmin (int): Tmin value.
            s_noise (int): Noise value.
            override_settings (dict): Dictionary of override settings.
            override_settings_restore_afterwards (bool): Flag to restore override settings afterwards.
            sampler: Sampler.
            include_init_images (bool): Include initial images flag.
            controlnet_units (List[ControlNetUnit]): List of control net units.
            use_deprecated_controlnet (bool): Use deprecated control net flag.
            outpath_samples (str): Output path for samples.
            sd_vae (str): VAE model checkpoint.
            sd_model_checkpoint (str): Model checkpoint.
            animatediff_flag (bool): Animatediff flag.
            animatediff_video_length (int): Animatediff video length.
            animatediff_fps (int): Animatediff video FPS.
            loractl_flag (bool): Whether to append LoRA weight in all steps to `gen_image` or not.

        Returns:
            gen_image (Union[PIL.Image.Image, List[PIL.Image.Image]]): Generated image.
        """
    if sampler is None:
        sampler = "Euler a"
    if steps is None:
        steps = 20
    # global tokenizer, scheduler, text_encoder, vae, unet, sd_model_checkpoint_before, pipeline
    # width = int(width // 8 * 8)
    # height = int(height // 8 * 8)

    # if (sd_model_checkpoint_before != sd_model_checkpoint) or (unet is None) or (vae is None) or (text_encoder is None):
    #     sd_model_checkpoint_before = sd_model_checkpoint
    #     print("load_models_from_stable_diffusion_checkpoint: start")
    #     text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(False, sd_model_checkpoint)
    #     print("load_models_from_stable_diffusion_checkpoint: over")

    # Load scheduler, tokenizer and models.
    # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_base15_checkpoint, subfolder="scheduler")
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     sd_base15_checkpoint, subfolder="tokenizer"
    # )

    # pipeline = StableDiffusionControlNetInpaintPipeline(
    #     controlnet=controlnet_units_list,
    #     unet=unet.to(weight_dtype),
    #     text_encoder=text_encoder.to(weight_dtype),
    #     vae=vae.to(weight_dtype),
    #     scheduler=noise_scheduler,
    #     tokenizer=tokenizer,
    #     safety_checker=None,
    #     feature_extractor=None,
    # ).to("cuda")
    # if preload_lora is not None:
    #     for _preload_lora in preload_lora:
    #         merge_lora(pipeline, _preload_lora, 0.60, from_safetensor=True, device="cuda", dtype=weight_dtype)
    # if len(sd_lora_checkpoint) != 0:
    #     # Bind LoRANetwork to pipeline.
    #     for _sd_lora_checkpoint in sd_lora_checkpoint:
    #         merge_lora(pipeline, _sd_lora_checkpoint, 0.90, from_safetensor=True, device="cuda", dtype=weight_dtype)

    # try:
    #     import xformers
    #     pipeline.enable_xformers_memory_efficient_attention()
    # except:
    #     logging.warning(
    #         'No module named xformers. Infer without using xformers. You can run pip install xformers to install it.')

    # generator = torch.Generator("cuda").manual_seed(int(seed))
    # pipeline.safety_checker = None

    # image = pipeline(
    #     prompt, image=images, mask_image=mask_image, control_image=controlnet_image, strength=denoising_strength,
    #     negative_prompt=negative_prompt,
    #     guidance_scale=cfg_scale, num_inference_steps=steps, generator=generator, height=height, width=width, \
    #     controlnet_conditioning_scale=controlnet_conditioning_scale, guess_mode=True
    # ).images[0]

    # if len(sd_lora_checkpoint) != 0:
    #     # Bind LoRANetwork to pipeline.
    #     for _sd_lora_checkpoint in sd_lora_checkpoint:
    #         unmerge_lora(pipeline, _sd_lora_checkpoint, 0.90, from_safetensor=True, device="cuda", dtype=weight_dtype)
    # if preload_lora is not None:
    #     for _preload_lora in preload_lora:
    #         unmerge_lora(pipeline, _preload_lora, 0.60, from_safetensor=True, device="cuda", dtype=weight_dtype)
    # return image

    # Pass sd_model to StableDiffusionProcessingTxt2Img does not work.
    # We should modify shared.opts.sd_model_checkpoint instead.
    p_img2img = StableDiffusionProcessingImg2Img(
        outpath_samples=outpath_samples,
        do_not_save_samples=do_not_save_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        sampler_name=sampler,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=images,
        mask=mask_image,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings,
        initial_noise_multiplier=initial_noise_multiplier,
    )
    p_img2img.scripts = scripts.scripts_img2img


def i2i_inpaint_call_old(
    images=[],
    mask_image=None,
    denoising_strength=0.75,
    controlnet_image=[],
    controlnet_units_list=[],
    controlnet_conditioning_scale=[],
    steps=20,
    seed=-1,

    cfg_scale=7.0,
    width=640,
    height=768,

    prompt="",
    negative_prompt="",
    sd_lora_checkpoint=[],
    sd_model_checkpoint="",
    sd_base15_checkpoint="",
):  
    global tokenizer, scheduler, text_encoder, vae, unet, sd_model_checkpoint_before, pipeline
    width = int(width // 8 * 8)
    height = int(height // 8 * 8)

    print("load_models_from_stable_diffusion_checkpoint: start")
    if (sd_model_checkpoint_before != sd_model_checkpoint) \
            or (unet is None) \
            or (vae is None) \
            or (text_encoder is None):
        sd_model_checkpoint_before = sd_model_checkpoint
        # TODO: 模型不匹配！！！！！
        text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(False, sd_model_checkpoint)
    print("load_models_from_stable_diffusion_checkpoint: over")

    # Load scheduler, tokenizer and models.
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_base15_checkpoint,
                                                                  subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        sd_base15_checkpoint, subfolder="tokenizer"
    )

    pipeline = StableDiffusionControlNetInpaintPipeline(
        controlnet=controlnet_units_list, 
        unet=unet.to(weight_dtype),
        text_encoder=text_encoder.to(weight_dtype),
        vae=vae.to(weight_dtype),
        scheduler=noise_scheduler,
        tokenizer=tokenizer,
        safety_checker=None,
        feature_extractor=None,
    ).to("cuda")
    if preload_lora is not None:
        for _preload_lora in preload_lora:
            merge_lora(pipeline, _preload_lora, 0.60, from_safetensor=True, device="cuda", dtype=weight_dtype)
    if len(sd_lora_checkpoint) != 0:
        # Bind LoRANetwork to pipeline.
        for _sd_lora_checkpoint in sd_lora_checkpoint:
            merge_lora(pipeline, _sd_lora_checkpoint, 0.90, from_safetensor=True, device="cuda", dtype=weight_dtype)

    try:
        import xformers
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        logging.warning('No module named xformers. '
                        'Infer without using xformers. You can run pip install xformers to install it.')
        
    generator           = torch.Generator("cuda").manual_seed(int(seed)) 
    pipeline.safety_checker = None

    image = pipeline(
        prompt,
        image=images,
        mask_image=mask_image,
        control_image=controlnet_image,
        strength=denoising_strength,
        negative_prompt=negative_prompt,
        guidance_scale=cfg_scale,
        num_inference_steps=steps,
        generator=generator,
        height=height,
        width=width,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guess_mode=True
    ).images[0]

    if len(sd_lora_checkpoint) != 0:
        # Bind LoRANetwork to pipeline.
        for _sd_lora_checkpoint in sd_lora_checkpoint:
            unmerge_lora(pipeline, _sd_lora_checkpoint, 0.90, from_safetensor=True, device="cuda", dtype=weight_dtype)
    if preload_lora is not None:
        for _preload_lora in preload_lora:
            unmerge_lora(pipeline, _preload_lora, 0.60, from_safetensor=True, device="cuda", dtype=weight_dtype)
    return image