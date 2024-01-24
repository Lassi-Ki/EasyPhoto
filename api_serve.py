import base64
import io
import hashlib
import boto3
import json

import torch.cuda
from fastapi import FastAPI
from easyphoto.easyphoto_train import *
from io import BytesIO


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(encoding)))
    return image


def encode_pil_to_base64(image):
    byte_io = io.BytesIO()
    image.save(byte_io, format='PNG')
    byte_data = byte_io.getvalue()
    base64_image = base64.b64encode(byte_data).decode()
    return base64_image


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/easyphoto/easyphoto_train_forward")
def _easyphoto_train_forward_api(datas: dict):
    sd_model_checkpoint = datas.get("sd_model_checkpoint", "sd_xl_base_1.0.safetensors")
    id_task = datas.get("id_task", "")
    user_id = datas.get("user_id", "tmp")
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

    # 传入的图片
    instance_images = datas.get("instance_images", [])
    validation = datas.get("validation", False)
    enable_rl = datas.get("enable_rl", False)
    max_rl_time = datas.get("max_rl_time", 1)
    timestep_fraction = datas.get("timestep_fraction", 1)

    args = datas.get("args", [])

    instance_images = [decode_base64_to_image(init_image) for init_image in instance_images]
    _instance_images = []
    for instance_image in instance_images:
        hash_value = hashlib.md5(instance_image.tobytes()).hexdigest()
        save_path = os.path.join('/tmp', hash_value + '.jpg')
        instance_image.save(save_path)
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
            *args
        )
    except Exception as e:
        torch.cuda.empty_cache()
        message = f"Train error, error info:{str(e)}"
    return {"message": message}
