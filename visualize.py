import argparse
import torch
import streamlit as st

from utils import get_one_image, get_file_length
from model.utils import load_llava

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datasets/mini_coco_2014/Images/", type=str)
    parser.add_argument("--model_name_or_path", default="llava-hf/llava-1.5-7b-hf", type=str)
    parser.add_argument("--quantization", action="store_true")
    
    args = parser.parse_args()
    return args

def set_up(args):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    args.device = device

    if args.device != "cuda":
        args.quantization = False

    args.model, args.tokenizer, args.processor = load_llava(model_name_or_path=args.model_name_or_path,
                                                            device_map="auto",
                                                            padding_side="left",
                                                            quantization = args.quantization)
    args.model.to(device)

def run_streamlit(args):
    st.set_page_config(page_title="Visualization", layout="wide")

    choose_img_container = st.container()
    with choose_img_container:
        choose_img_col1, _, choose_img_col2 = st.columns([2,1,3])
        # Select image index,
        # if index = -1, randomly select one image from the folder
        # 15133: man, horse
        num_image_files = get_file_length()
        with choose_img_col1:
            image_idx = st.text_input(label=f"Select an index from {num_image_files} images",
                                    value="15133")
            img_np = get_one_image(idx=int(image_idx), image_path=args.data_dir)
            st.write(f"Select index {image_idx} from {num_image_files} images")

        with choose_img_col2:
            st.image(img_np)

if __name__ == "__main__":
    args = parse_args()
    set_up(args)
    run_streamlit(args)