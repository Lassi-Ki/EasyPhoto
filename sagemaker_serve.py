import boto3
import torch
import os
import base64
import numpy as np
from easyphoto.easyphoto_config import *
import hashlib
from PIL import Image
from easyphoto.easyphoto_train import easyphoto_train_forward
from easyphoto.easyphoto_infer import easyphoto_infer_forward
from io import BytesIO


def open_image(image_path):
    image = Image.open(image_path)
    return image


def download_from_s3(bucket_name, s3_folder, local_folder):
    s3 = boto3.client('s3',
                      region_name=AWS_REGION,
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    # 列出 S3 文件夹下的所有对象
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)['Contents']

    # 遍历所有对象并下载到本地文件夹
    for obj in objects:
        # 构建本地文件路径
        local_file_path = os.path.join(local_folder, os.path.relpath(obj['Key'], s3_folder))
        # 创建本地文件夹（如果不存在）
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        # 下载文件
        s3.download_file(bucket_name, obj['Key'], local_file_path)
        print(f"Downloaded: {obj['Key']} to {local_file_path}")


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(encoding)))
    return image


def train(datas: dict):
    user_id = datas.get("user_id", "tmp")
    images_s3_path = datas.get("images_s3_path", "")
    """------------------------------- 固定参数 ---------------------------------------"""
    sd_model_checkpoint = datas.get("sd_model_checkpoint", "sd_xl_base_1.0.safetensors")
    resolution = datas.get("resolution", 1024)
    val_and_checkpointing_steps = datas.get("val_and_checkpointing_steps", 100)
    max_train_steps = datas.get("max_train_steps", 600)
    steps_per_photos = datas.get("steps_per_photos", 200)
    train_batch_size = datas.get("train_batch_size", 1)
    gradient_accumulation_steps = datas.get("gradient_accumulation_steps", 4)
    dataloader_num_workers = datas.get("dataloader_num_workers", 16)
    learning_rate = datas.get("learning_rate", 1e-4)
    rank = datas.get("rank", 32)
    network_alpha = datas.get("network_alpha", 16)
    validation = datas.get("validation", False)
    enable_rl = datas.get("enable_rl", False)
    max_rl_time = datas.get("max_rl_time", 1)
    timestep_fraction = datas.get("timestep_fraction", 1)
    id_task = datas.get("id_task", "")
    skin_retouching_bool = datas.get("skin_retouching_bool", True)
    args = datas.get("args", [])

    current_directory = os.getcwd()
    folder_path = current_directory + '/data'
    folder_tmp_path = current_directory + '/tmp'
    print(folder_path)

    # TODO: 后期转为 input channel 输入
    #download_from_s3(AWS_S3_BUCKET_NAME, images_s3_path, folder_path)

    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    instance_images = [open_image(file) for file in files]
    _instance_images = []

    for instance_image in instance_images:
        hash_value = hashlib.md5(instance_image.tobytes()).hexdigest()
        save_path = os.path.join(folder_tmp_path, hash_value + '.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        instance_image_rgb = instance_image.convert("RGB")
        instance_image_rgb.save(save_path)
        _instance_images.append({"name": save_path})
    instance_images = _instance_images

    try:
        message = easyphoto_train_forward(
            sd_model_checkpoint,
            id_task,
            user_id,
            resolution,
            val_and_checkpointing_steps,
            max_train_steps,
            steps_per_photos,
            train_batch_size,
            gradient_accumulation_steps,
            dataloader_num_workers,
            learning_rate,
            rank,
            network_alpha,
            validation,
            instance_images,
            enable_rl,
            max_rl_time,
            timestep_fraction,
            skin_retouching_bool,
            *args
        )
    except Exception as e:
        torch.cuda.empty_cache()
        message = f"Train error, error info:{str(e)}"
    return {"message": message}


def inference(datas: dict):
    user_ids = datas.get("user_ids", [])
    init_image = datas.get("init_image", None)
    selected_template_images = datas.get("selected_template_images", [])
    uploaded_template_images = datas.get("uploaded_template_images", [])
    tabs = datas.get("tabs", 0)
    # ------------------------------------------------------------------------------------------
    sd_model_checkpoint = datas.get("sd_model_checkpoint", "sd_xl_base_1.0.safetensors")
    additional_prompt = datas.get("additional_prompt", "")
    # ------------------------------------------------------------------------------------------
    first_diffusion_steps = datas.get("first_diffusion_steps", 50)
    first_denoising_strength = datas.get("first_denoising_strength", 0.45)
    second_diffusion_steps = datas.get("second_diffusion_steps", 20)
    second_denoising_strength = datas.get("second_denoising_strength", 0.35)
    seed = datas.get("seed", -1)
    crop_face_preprocess = datas.get("crop_face_preprocess", True)
    before_face_fusion_ratio = datas.get("before_face_fusion_ratio", 0.50)
    after_face_fusion_ratio = datas.get("after_face_fusion_ratio", 0.50)
    apply_face_fusion_before = datas.get("apply_face_fusion_before", True)
    apply_face_fusion_after = datas.get("apply_face_fusion_after", True)
    color_shift_middle = datas.get("color_shift_middle", True)
    color_shift_last = datas.get("color_shift_last", True)
    super_resolution = datas.get("super_resolution", True)
    display_score = datas.get("display_score", False)
    background_restore = datas.get("background_restore", False)
    background_restore_denoising_strength = datas.get("background_restore", 0.35)
    sd_xl_input_prompt = datas.get("sd_xl_input_prompt",
                                   "upper-body, look at viewer, one twenty years old girl, wear white shit, standing, in the garden, daytime, f32")
    sd_xl_resolution = datas.get("sd_xl_resolution", "(1024, 1024)")

    if type(user_ids) == str:
        user_ids = [user_ids]

    current_directory = os.getcwd()
    folder_path = current_directory + '/model_data/infer_templates'
    print("infer_templates path: ", folder_path)

    selected_template_images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, file))]
    selected_template_images = str(selected_template_images)
    # selected_template_images = [open_image(file) for file in files]

    # init_image = None if init_image is None else decode_base64_to_image(init_image)
    # selected_template_images = [decode_base64_to_image(_) for _ in selected_template_images]
    # uploaded_template_images = [decode_base64_to_image(_) for _ in uploaded_template_images]

    if init_image is not None:
        init_image = np.uint8(init_image)

    tabs = int(tabs)
    try:
        comment, outputs, face_id_outputs = easyphoto_infer_forward(
            sd_model_checkpoint, selected_template_images, init_image, uploaded_template_images, additional_prompt, \
            before_face_fusion_ratio, after_face_fusion_ratio, first_diffusion_steps, first_denoising_strength,
            second_diffusion_steps, second_denoising_strength, \
            seed, crop_face_preprocess, apply_face_fusion_before, apply_face_fusion_after, color_shift_middle,
            color_shift_last, super_resolution, display_score, background_restore, \
            background_restore_denoising_strength, sd_xl_input_prompt, sd_xl_resolution, tabs, *user_ids
        )
        # outputs = [encode_pil_to_base64(output) for output in outputs]
    except Exception as e:
        torch.cuda.empty_cache()
        comment = f"Infer error, error info:{str(e)}"
        outputs = []
        face_id_outputs = []

    return {"message": comment, "outputs": outputs, "face_id_outputs": face_id_outputs}