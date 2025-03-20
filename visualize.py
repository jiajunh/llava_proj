import argparse
import torch
import asyncio
import streamlit as st

import requests
import numpy as np
from PIL import Image

from utils import get_one_image, get_file_length
from model.utils import load_llava, get_llava_image_features
from logit_lens.generator import LogitLens
from logit_lens.display import LogitLensVisualizer


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

    # args.data_dir = "datasets/mini_coco_2014/Images/"
    args.data_dir = "/kaggle/input/mini-coco2014-dataset-for-image-captioning/Images/"

    if args.device != "cuda":
        args.quantization = False

    args.model, args.tokenizer, args.processor = load_llava(model_name_or_path=args.model_name_or_path,
                                                            device_map="auto",
                                                            padding_side="left",
                                                            quantization = args.quantization)
    
    args.lv = LogitLensVisualizer(patch_size=14, image_size=336)
    args.generator = LogitLens(args.model, args.processor, args.tokenizer)

    
@st.cache_data
def get_logit_lens_test_img():
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    img_np = np.asarray(image)
    return img_np

@st.cache_data
def logit_lens_visualize(_args, patch_topk=20, k_most_freq=100):
    img_np = get_logit_lens_test_img()
    filtered_tokens = None,
    next_five_tokens = None,
    # image_features = get_llava_image_features(_args.model, _args.processor, img_np)
    # next_five_tokens = _args.generator.batch_generate(img_np)
    # next_tokens_ids = _args.generator.get_generated_ids(img_np, topk=patch_topk)
    # most_freq_token_ids = _args.generator.get_most_frequent_token_ids(next_tokens_ids, k=k_most_freq)
    # most_freq_tokens = _args.generator.decode(most_freq_token_ids)
    # filtered_tokens = _args.generator.filter_tokens(most_freq_tokens)
    
    return {"image": img_np,
            "filtered_tokens": filtered_tokens,
            "next_five_tokens": next_five_tokens,
            }



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


    logit_lens_container = st.container()
    with logit_lens_container:
        st.write("Logit lens part")
        img_col, text_col, token_on_img_col = st.columns([2,1,2])

        lv_result = logit_lens_visualize(args, patch_topk=20, k_most_freq=100)

        with img_col:
            img_np = lv_result["image"]
            st.image(img_np)

        with text_col:
            patch_topk = st.text_input(label=f"patch_topk",
                                    value="20")
            k_most_freq = st.text_input(label=f"k_most_freq",
                                    value="100")
            lv_result = logit_lens_visualize(args, 
                                      patch_topk=int(patch_topk), 
                                      k_most_freq=int(k_most_freq))
            st.write(f"filtered_tokens: {lv_result['filtered_tokens']}")

        with token_on_img_col:
            selected_token = st.text_input(label=f"Choose a token",
                                    value="▁sign")
            
            # mask = args.generator.patch_with_given_token(lv_result["image"], selected_token, topk=50)
            
            # fig = args.lv.plot_tokens_on_image(img_np, tokens=lv_result["next_five_tokens"], 
            #                         show_full_image=False, 
            #                         part_idx=0,
            #                         n_splits=4,
            #                         use_resized_img=False,
            #                         text_fontsize=14)
            # st.pyplot(fig)



    

    

if __name__ == "__main__":
    args = parse_args()
    set_up(args)
    run_streamlit(args)