import os
import platform
import subprocess
import sys
import numpy as np
from glob import glob
from easyphoto.sd_models_config import config_sdxl
from PIL import Image, ImageOps
from easyphoto.easyphoto_config import (cache_log_file_path,
                                        models_path,
                                        user_id_outpath_samples,
                                        validation_prompt)
from easyphoto.easyphoto_utils import check_id_valid

python_executable_path = sys.executable
check_hash = True


def easyphoto_train_forward(
    sd_model_checkpoint: str,
    id_task: str,
    user_id: str,
    resolution: int,
    val_and_checkpointing_steps: int,
    max_train_steps: int,
    steps_per_photos: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int,
    learning_rate: float,
    rank: int, network_alpha: int,
    validation: bool,
    instance_images: list,
    enable_rl: bool,
    max_rl_time: float,
    timestep_fraction: float,
    skin_retouching_bool: bool,
    *args
):  
    global check_hash

    if user_id == "" or user_id is None:
        return "User id cannot be set to empty."
    if user_id == "none":
        return "User id cannot be set to none."

    ids = []
    if os.path.exists(user_id_outpath_samples):
        _ids = os.listdir(user_id_outpath_samples)
        for _id in _ids:
            if check_id_valid(_id, user_id_outpath_samples, models_path):
                ids.append(_id)
    ids = sorted(ids)
    print(f"ids: {len(ids)}")
    if user_id in ids:
        return "User id non-repeatability."

    if int(rank) < int(network_alpha):
        return "The network alpha {} must not exceed rank {}. " "It will result in an unintended LoRA.".format(network_alpha, rank)

    # check_files_exists_and_download(check_hash.get("sdxl", True), "sdxl")
    # check_hash = False

    if int(resolution) < 1024:
        return "The resolution for SDXL Training needs to be 1024."
    if validation:
        # We do not ensemble models by validation in SDXL training.
        return "To save training time and VRAM, please turn off validation in SDXL training."
    
    # training templates path
    training_templates_path = os.path.join(models_path, "training_templates")
    # origin image copy
    original_backup_path = os.path.join(user_id_outpath_samples, user_id, "original_backup")
    # ref image copy
    ref_image_path = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

    # training data save path
    user_path = os.path.join(user_id_outpath_samples, user_id, "processed_images")
    images_save_path = os.path.join(user_id_outpath_samples, user_id, "processed_images", "train")
    json_save_path = os.path.join(user_id_outpath_samples, user_id, "processed_images", "metadata.jsonl")

    # weight save path
    weights_save_path = os.path.join(user_id_outpath_samples, user_id, "user_weights")
    webui_save_path = os.path.join(models_path, f"Lora/{user_id}.safetensors")
    webui_load_path = os.path.join(models_path, f"Others/stable-diffusion-xl", sd_model_checkpoint)
    sd_save_path = os.path.join(models_path, "stable-diffusion-xl/stabilityai_stable_diffusion_xl_base_1.0")
    
    os.makedirs(original_backup_path, exist_ok=True)
    os.makedirs(user_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(webui_save_path, exist_ok=True)
    os.makedirs(sd_save_path, exist_ok=True)

    max_train_steps = int(min(len(instance_images) * int(steps_per_photos), int(max_train_steps)))

    for index, user_image in enumerate(instance_images):
        image = Image.open(user_image['name'])
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(os.path.join(original_backup_path, str(index) + ".jpg"))

    # preprocess
    preprocess_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess.py")
    command = [
            f'{python_executable_path}',
            f'{preprocess_path}',
            f'--images_save_path={images_save_path}',
            f'--json_save_path={json_save_path}', 
            f'--validation_prompt={validation_prompt}',
            f'--inputs_dir={original_backup_path}',
            f'--ref_image_path={ref_image_path}'
        ]
    if skin_retouching_bool:
        command += ["--skin_retouching_bool"]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the command: {e}")
        
    # check preprocess results
    train_images = glob(os.path.join(images_save_path, "*.jpg"))
    if len(train_images) == 0:
        return "Failed to obtain preprocessed images, please check the preprocessing process"
    if not os.path.exists(json_save_path):
        return "Failed to obtain preprocessed metadata.jsonl, please check the preprocessing process."

    train_kohya_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "train_kohya/train_lora_sd_XL.py")
    
    # extensions/sd-webui-EasyPhoto/train_kohya_log.txt, use to cache log and flush to UI
    print("cache_log_file_path:", cache_log_file_path)
    if not os.path.exists(os.path.dirname(cache_log_file_path)):
        os.makedirs(os.path.dirname(cache_log_file_path), exist_ok=True)

    # Extra arguments to run SDXL training.
    env = None
    original_config = config_sdxl
    sdxl_model_dir = os.path.join(models_path, "stable-diffusion-xl")
    pretrained_vae_model_name_or_path = os.path.join(sdxl_model_dir, "madebyollin_sdxl_vae_fp16_fix")
    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TRANSFORMERS_CACHE"] = sdxl_model_dir
    # message = unload_models()
    # print(message)
    random_seed = np.random.randint(1, 1e6)
    if platform.system() == 'Windows':
        print("Can not Windows!")
        pass
    else:
        command = [
            f'{python_executable_path}',
            '-m',
            'accelerate.commands.launch',
            '--mixed_precision=fp16',
            "--main_process_port=3456",
            f'{train_kohya_path}',
            f'--pretrained_model_name_or_path={sd_save_path}',
            f'--pretrained_model_ckpt={webui_load_path}', 
            f'--train_data_dir={user_path}',
            '--caption_column=text', 
            f'--resolution={resolution}',
            '--random_flip',
            f'--train_batch_size={train_batch_size}',
            f'--gradient_accumulation_steps={gradient_accumulation_steps}',
            f'--dataloader_num_workers={dataloader_num_workers}', 
            f'--max_train_steps={max_train_steps}',
            f'--checkpointing_steps={val_and_checkpointing_steps}', 
            f'--learning_rate={learning_rate}',
            '--lr_scheduler=constant',
            '--lr_warmup_steps=0', 
            '--train_text_encoder', 
            f'--seed={random_seed}',
            f'--rank={rank}',
            f'--network_alpha={network_alpha}', 
            f'--validation_prompt={validation_prompt}', 
            f'--validation_steps={val_and_checkpointing_steps}', 
            f'--output_dir={weights_save_path}', 
            f'--logging_dir={weights_save_path}', 
            '--enable_xformers_memory_efficient_attention', 
            '--mixed_precision=fp16', 
            f'--template_dir={training_templates_path}', 
            '--template_mask', 
            '--merge_best_lora_based_face_id', 
            f'--merge_best_lora_name={user_id}',
            f'--cache_log_file={cache_log_file_path}'
        ]
        if validation:
            command += ["--validation"]
        command += [f"--original_config={original_config}"]
        command += [f"--pretrained_vae_model_name_or_path={pretrained_vae_model_name_or_path}"]
        try:
            print("Working...")
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")
    
    best_weight_path = os.path.join(weights_save_path, f"best_outputs/{user_id}.safetensors")
    # Currently, SDXL training doesn't support the model selection and ensemble. We use the final
    # trained model as the best for simplicity.
    best_weight_path = os.path.join(weights_save_path, "pytorch_lora_weights.safetensors")
    if not os.path.exists(best_weight_path):
        return "Failed to obtain Lora after training, please check the training process."
    # copyfile(best_weight_path, webui_save_path)
    return "The training has been completed."
