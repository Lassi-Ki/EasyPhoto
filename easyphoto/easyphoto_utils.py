import logging
import os
import time
import hashlib
import requests
from modelscope import snapshot_download
from easyphoto.easyphoto_config import data_path, models_path, script_path
import easyphoto
import torch
import gc

download_urls = {
    # The models are from civitai/6424 & civitai/118913, we saved them to oss for your convenience in downloading the models.
    "sdxl": [
        # sdxl
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/diffusers_xl_canny_mid.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/thibaud_xl_openpose_256lora.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/madebyollin_sdxl_vae_fp16_fix/diffusion_pytorch_model.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/madebyollin-sdxl-vae-fp16-fix.safetensors",
    ],
}

save_filenames = {
    # The models are from civitai/6424 & civitai/118913, we saved them to oss for your convenience in downloading the models.
    "sdxl": [
        [
            os.path.join(models_path, f"ControlNet/diffusers_xl_canny_mid.safetensors"),
            os.path.join(controlnet_cache_path, f"models/diffusers_xl_canny_mid.safetensors"),
        ],
        [
            os.path.join(models_path, f"ControlNet/thibaud_xl_openpose_256lora.safetensors"),
            os.path.join(controlnet_cache_path, f"models/thibaud_xl_openpose_256lora.safetensors"),
        ],
        os.path.join(models_path, "stable-diffusion-xl/madebyollin_sdxl_vae_fp16_fix/diffusion_pytorch_model.safetensors"),
        os.path.join(models_path, f"VAE/madebyollin-sdxl-vae-fp16-fix.safetensors"),
    ],
}

# Set the level of the logger
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.getLogger().setLevel(log_level)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')  

def save_image(image, path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    index = len([path for path in os.listdir(path) if path.endswith('.jpg')]) + 1
    image_path = os.path.join(path, str(index).zfill(8) + '.jpg')
    return image.save(image_path)

def check_id_valid(user_id, user_id_outpath_samples, models_path):
    face_id_image_path = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg") 
    print(face_id_image_path)
    if not os.path.exists(face_id_image_path):
        return False
    
    safetensors_lora_path   = os.path.join(user_id_outpath_samples, user_id, "user_weights", "best_outputs", f"{user_id}.safetensors") 
    print(safetensors_lora_path)
    if not os.path.exists(safetensors_lora_path):
        return False
    return True


def urldownload_progressbar(url, filepath):
    start = time.time() 
    response = requests.get(url, stream=True)
    size = 0 
    chunk_size = 1024
    content_size = int(response.headers['content-length']) 
    try:
        if response.status_code == 200: 
            print('Start download,[File size]:{size:.2f} MB'.format(size = content_size / chunk_size /1024))  
            with open(filepath,'wb') as file:  
                for data in response.iter_content(chunk_size = chunk_size):
                    file.write(data)
                    size +=len(data)
                    print('\r'+'[下载进度]:%s%.2f%%' % ('>'*int(size*50/ content_size), float(size / content_size * 100)) ,end=' ')
        end = time.time()
        print('Download completed!,times: %.2f秒' % (end - start))
    except:
        print('Error!')


# TODO: 将所有需要下载的文件放入S3桶中，后续通过S3桶进行下载
def check_files_exists_and_download(check_hash, download_mode="base"):
    urls, filenames = download_urls[download_mode], save_filenames[download_mode]
    for url, filename in zip(urls, filenames):
        if type(filename) is str:
            filename = [filename]

        exist_flag = False
        for _filename in filename:
            if not check_hash:
                if os.path.exists(_filename):
                    exist_flag = True
                    break
            else:
                if os.path.exists(_filename):
                    exist_flag = True
                    break
        if exist_flag:
            continue

        print(f"Start Downloading: {url}")
        os.makedirs(os.path.dirname(filename[0]), exist_ok=True)
        urldownload_progressbar(url, filename[0])


# Calculate the hash value of the download link and downloaded_file by sha256
def compare_hasd_link_file(url, file_path):
    r           = requests.head(url)
    total_size  = int(r.headers['Content-Length'])
    
    res = requests.get(url, stream=True)
    remote_head_hash = hashlib.sha256(res.raw.read(1000)).hexdigest()  
    res.close()
    
    end_pos = total_size - 1000
    headers = {'Range': f'bytes={end_pos}-{total_size-1}'}
    res = requests.get(url, headers=headers, stream=True)
    remote_end_hash = hashlib.sha256(res.content).hexdigest()
    res.close()
    
    with open(file_path,'rb') as f:
        local_head_data = f.read(1000)
        local_head_hash = hashlib.sha256(local_head_data).hexdigest()
    
        f.seek(end_pos)
        local_end_data = f.read(1000) 
        local_end_hash = hashlib.sha256(local_end_data).hexdigest()
     
    if remote_head_hash == local_head_hash and remote_end_hash == local_end_hash:
        print(f"{file_path} : Hash match")
        return True
      
    else:
        print(f" {file_path} : Hash mismatch")
        return False


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'

def css_html():
    head = ""

    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    for cssfile in ["style.css"]:
        if not os.path.isfile(cssfile):
            continue

        head += stylesheet(cssfile)

    if os.path.exists(os.path.join(data_path, "user.css")):
        head += stylesheet(os.path.join(data_path, "user.css"))

    return head

def unload_models():
    """Unload models to free VRAM."""
    easyphoto.easyphoto_infer.retinaface_detection = None
    easyphoto.easyphoto_infer.image_face_fusion = None
    easyphoto.easyphoto_infer.skin_retouching = None
    easyphoto.easyphoto_infer.portrait_enhancement = None
    easyphoto.easyphoto_infer.face_skin = None
    easyphoto.easyphoto_infer.face_recognition = None
    easyphoto.easyphoto_infer.psgan_inference = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return "Already Empty Cache of Preprocess Model in EasyPhoto"
