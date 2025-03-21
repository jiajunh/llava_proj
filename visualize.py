import argparse
import torch
import streamlit as st

import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import get_one_image, get_file_length
from model.utils import load_llava, get_llava_image_features, get_llava_inputs_outputs
from logit_lens.generator import LogitLens
from logit_lens.display import LogitLensVisualizer
from attentions.attns import AttentionGenerator
from attentions.display import AttentionVisualizer


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

    args.data_dir = "/kaggle/input/mini-coco2014-dataset-for-image-captioning/Images/"

    if args.device != "cuda":
        args.quantization = False

    args.model, args.tokenizer, args.processor = load_llava(model_name_or_path=args.model_name_or_path,
                                                            device_map="auto",
                                                            padding_side="left",
                                                            quantization = args.quantization)
    
    args.lv = LogitLensVisualizer(patch_size=14, image_size=336)
    args.generator = LogitLens(args.model, args.processor, args.tokenizer)
    args.ag = AttentionGenerator(args.model, args.processor, args.tokenizer)
    args.vis = AttentionVisualizer(patch_size=14, image_size=336)

    args.generate_config = {
        "max_new_tokens": 50,
        # "num_beams": 3,
        # "early_stopping": True,
        "do_sample": False,
        # "top_p": 0.3,
        "return_dict_in_generate": True,
        "output_attentions": True,
        "output_hidden_states": True,
    }

    
@st.cache_data
def get_logit_lens_test_img():
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    img_np = np.asarray(image)
    return img_np


def logit_lens_visualize(args, patch_topk=20, k_most_freq=100):
    img_np = st.session_state["img_np"]
    filtered_tokens = None,
    next_five_tokens = None,
    image_features = get_llava_image_features(args.model, args.processor, img_np)
    next_five_tokens = args.generator.batch_generate(img_np)
    next_tokens_ids = args.generator.get_generated_ids(img_np, topk=patch_topk)
    most_freq_token_ids = args.generator.get_most_frequent_token_ids(next_tokens_ids, k=k_most_freq)
    most_freq_tokens = args.generator.decode(most_freq_token_ids)
    filtered_tokens = args.generator.filter_tokens(most_freq_tokens)

    st.session_state["image_features"] = image_features
    st.session_state["filtered_tokens"] = filtered_tokens
    st.session_state["next_five_tokens"] = next_five_tokens


@st.fragment
def st_select_image_container(args):
    print("-"*10, "Run select image fragment", "-"*10)
    choose_img_container = st.container()
    choose_img_container.header("Choose image")
    with choose_img_container:
        choose_img_col1, _, choose_img_col2 = st.columns([2,1,3])
        # Select image index,
        # if index = -1, randomly select one image from the folder
        # 15133: man, horse
        num_image_files = get_file_length(path=args.data_dir)
        with choose_img_col1:
            image_idx = st.text_input(label=f"Select an index from {num_image_files} images",
                                    value="15133")
            img_np = get_one_image(idx=int(image_idx), image_path=args.data_dir)
            st.session_state["img_np"] = img_np
            st.session_state["img_idx"] = image_idx
            st.write(f"Select index {image_idx} from {num_image_files} images")
        with choose_img_col2:
            st.image(st.session_state["img_np"])


@st.fragment
def st_logit_lens_container(args):
    print("-"*10, "Run logit lens fragment", "-"*10)
    logit_lens_container = st.container()
    logit_lens_container.header("Logit lens")
    with logit_lens_container:
        input_col, token_on_img_col, salicy_map_col = st.columns([1,2,2])

        with input_col:
            with st.form("logit lens inputs"):
                patch_topk = st.text_input(label=f"patch_topk", value="20")
                k_most_freq = st.text_input(label=f"k_most_freq", value="100")

                freq_token_submitted = st.form_submit_button("freq tokens")
                if freq_token_submitted:
                    logit_lens_visualize(args, patch_topk=int(patch_topk), k_most_freq=int(k_most_freq))
                
                if "filtered_tokens" in st.session_state:
                    st.write(f"filtered_tokens: {st.session_state['filtered_tokens']}")
                else:
                    st.write("filtered_tokens: None")

                selected_token = st.text_input(label=f"Choose a token", value="")
                st.session_state["selected_token"] = selected_token
            
        with token_on_img_col:
            if not "filtered_tokens" in st.session_state:
                st.write("No data")
            else:
                fig = args.lv.plot_tokens_on_image(image=st.session_state["img_np"], 
                                             tokens=st.session_state["next_five_tokens"], 
                                             show_full_image=False, 
                                             part_idx=0,
                                             n_splits=4,
                                             use_resized_img=False,
                                             text_fontsize=10)
                st.pyplot(fig)

        with salicy_map_col:
            if not st.session_state["selected_token"]:
                st.write("No data")
            else:
                mask = args.generator.patch_with_given_token(image=st.session_state["img_np"],
                                                            input_token=st.session_state["selected_token"], 
                                                            topk=50)
                fig = args.lv.plot_saliency_map(image=st.session_state["img_np"], 
                                                mask=mask)
                st.pyplot(fig)

@st.cache_data
def st_generate(_args, img):
    inputs, outputs = get_llava_inputs_outputs(image=img, 
                                               model=_args.model, 
                                               processor = _args.processor, 
                                               generate_config=_args.generate_config)
    st.session_state["inputs"] = inputs
    st.session_state["outputs"] = outputs

@st.fragment
def st_attention_maps(args):
    print("-"*10, "Run attention map fragment", "-"*10)
    st_generate(args, st.session_state["img_np"])

    attention_map_container = st.container()
    attention_map_container.header("Attention maps")

    generated_sequences = args.processor.batch_decode(st.session_state["outputs"]["sequences"], 
                                                      skip_special_tokens=True, 
                                                      clean_up_tokenization_spaces=False)
    modified_token_ids, modified_token_list = args.ag.decode_tokens(st.session_state["inputs"], st.session_state["outputs"])

    with attention_map_container:
        text_col, attention_map_col = st.columns([1,4])

        with text_col:
            st.write(f"Generated sequence: \n {generated_sequences} \n")
            st.write(f"Generated tokens: \n {modified_token_list} \n")
            
            selected_token = st.text_input(label=f"select a token", value="")

        with attention_map_col:
            pass
    


def run_streamlit(args):
    st.set_page_config(page_title="Visualization", layout="wide")
    # Load image part
    st_select_image_container(args)
    # Logit lens part
    st_logit_lens_container(args)
    # Attention maps
    st_attention_maps(args)

    

    

if __name__ == "__main__":
    args = parse_args()
    set_up(args)
    run_streamlit(args)