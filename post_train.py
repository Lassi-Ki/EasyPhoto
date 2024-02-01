import base64
import json
import os
import sys
import time
from glob import glob
import requests
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./data/")
parser.add_argument("--userID", type=str, default="nn")
cmd_opts = parser.parse_args()


def post_train(encoded_images, userID, url='http://0.0.0.0:7860'):
    datas = json.dumps({
        "user_id"               : userID, # A custom ID that identifies the trained face model
        "sd_model_checkpoint"   : "SDXL_1.0_ArienMixXL_v2.0.safetensors",
        "resolution"            : 1024,
        "val_and_checkpointing_steps" : 100,
        "max_train_steps"       : 600, # Training steps
        "steps_per_photos"      : 200,
        "train_batch_size"      : 1,
        "gradient_accumulation_steps" : 4,
        "dataloader_num_workers" : 16,
        "learning_rate"         : 1e-4,
        "rank"                  : 128,
        "network_alpha"         : 64,
        "instance_images"       : encoded_images, 
    })
    r = requests.post(f'{url}/easyphoto/easyphoto_train_forward', data=datas, timeout=1500)
    data = r.content.decode('utf-8')
    return data


if __name__ == '__main__':
    # initiate time
    time_start = time.time()  
    
    # -------------------training procedure------------------- #
    # When selecting a folder as a parameter input: ./data/
    img_list = cmd_opts.path
    encoded_images = []
    for idx, img_path in enumerate(img_list):
        with open(img_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')
            encoded_images.append(encoded_image)
    outputs = post_train(encoded_images, cmd_opts.userID)
    outputs = json.loads(outputs)
    print(outputs['message'])
    
    # End of record time
    # The calculated time difference is the execution time of the program, expressed in minute / m
    time_end = time.time()  
    time_sum = (time_end - time_start) // 60  
    
    print('# --------------------------------------------------------- #')
    print(f"#   Total expenditure：{time_sum} minutes ")
    print('# --------------------------------------------------------- #')