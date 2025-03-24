import os
import io
import random
import base64
import numpy as np
from PIL import Image
from typing import Union


def get_file_length(path="datasets/mini_coco_2014/Images/"):
    files = os.listdir(path)
    return len(files)

def get_one_image(idx=-1,
                  image_path="datasets/mini_coco_2014/Images/")-> np.ndarray:
    image_files = sorted(os.listdir(image_path))
    data_size = len(image_files)
    assert(idx >=- 1 and idx < data_size)
    if idx < 0:
        idx = random.randint(1, data_size)-1
    # print(f"choose index {idx} from total {data_size} images")
    img_path = image_path + image_files[idx]
    image = Image.open(img_path)
    np_image = np.asarray(image)
    return np_image

def image_to_base64(img: Union[np.ndarray, Image.Image]):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def blank_heatmap_on_img(img, alpha=0.4):
    img = img.astype(np.float32) / 255.0
    rgb_color = np.array([0, 0, 230], dtype=np.float32) / 255
    heat_map = np.ones(img.shape) * rgb_color
    overlayed_img = img * (1-alpha) + heat_map * alpha
    return (overlayed_img * 255).astype(np.uint8)



