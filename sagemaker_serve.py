import boto3
import torch
import os
from easyphoto.easyphoto_config import *
import hashlib
from PIL import Image
from easyphoto.easyphoto_train import easyphoto_train_forward


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
