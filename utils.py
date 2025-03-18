import os
import random

def get_one_image(idx=-1,
                  image_path="datasets/mini_coco_2014/Images/"):
    image_files = os.listdir(image_path)
    data_size = len(image_files)
    assert(idx >=- 1 and idx < data_size)
    if idx < 0:
        idx = random.randint(1, data_size)-1
    print(f"choose index {idx} from total {data_size} images")
    img_path = image_path + image_files[idx]
    return img_path
