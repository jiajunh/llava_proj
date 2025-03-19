import torch
import requests
import numpy as np
from PIL import Image

from utils import get_one_image, get_file_length
from model.utils import load_llava, get_llava_image_features, get_llava_logits
from logit_lens.generator import LogitLens


def get_logit_lens_test_img():
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    img_np = np.asarray(image)
    return img_np

def test():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    model, tokenizer, processor = load_llava(model_name_or_path="llava-hf/llava-1.5-7b-hf",
                                            device_map="auto",
                                            padding_side="left",
                                            quantization = True)

    img_np = get_logit_lens_test_img()
    print("-"*20, "Start testing", "-"*20)
    print("Load image: ", img_np.shape)

    generator = LogitLens(model, processor, tokenizer)
    image_features = get_llava_image_features(model, processor, img_np)
    print("One image feature shape", image_features.shape)

    next_five_token_ids = generator.batch_generate(img_np)
    print("topk next token for each patch: ", len(next_five_token_ids), next_five_token_ids[0])

    next_tokens_ids = generator.get_generated_ids(img_np, topk=20)
    most_freq_token_ids = generator.get_most_frequent_token_ids(next_tokens_ids, k=100)
    most_freq_tokens = generator.decode(most_freq_token_ids)
    print("Most frequent tokens ids: ", most_freq_tokens)

    filtered_tokens = generator.filter_tokens(most_freq_tokens)
    print("Filtered tokens: ", filtered_tokens)

if __name__ == "__main__":
    test()