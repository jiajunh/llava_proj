import torch
import requests
import numpy as np
from PIL import Image

from utils import get_one_image, get_file_length
from model.utils import load_llava, get_llava_image_features, get_llava_logits, get_llava_inputs_outputs
from logit_lens.generator import LogitLens
from attentions.attns import AttentionGenerator
from attentions.display import AttentionVisualizer


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
    print(type(img_np))
    print("-"*20, "Start testing", "-"*20)
    print("-"*20, "test logit lens" , "-"*20)

    print("Load image: ", img_np.shape)

    generator = LogitLens(model, processor, tokenizer)
    image_features = get_llava_image_features(model, processor, img_np)
    print("One image feature shape", image_features.shape)

    next_five_token_ids = generator.batch_generate(img_np)
    print("topk next token for each patch: ", len(next_five_token_ids), next_five_token_ids[0])

    next_tokens_ids = generator.get_generated_ids(img_np, topk=20)
    most_freq_token_ids = generator.get_most_frequent_token_ids(next_tokens_ids, k=50)
    most_freq_tokens = generator.decode(most_freq_token_ids)
    print("Most frequent tokens ids: ", most_freq_tokens)

    filtered_tokens = generator.filter_tokens(most_freq_tokens)
    print("Filtered tokens: ", filtered_tokens)

    mask = generator.patch_with_given_token(img_np, "▁sign", topk=50)
    print("Patch mask: ", mask)


    print("-"*20, "test attention scores" , "-"*20)
    img_np = get_one_image(idx=15133)
    inputs, outputs = get_llava_inputs_outputs(img_np, model, processor)
    generate_config = {
        "max_new_tokens": 50,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_attentions": True,
        "output_hidden_states": True,
    }
    ag = AttentionGenerator(model, processor, tokenizer)
    inputs, outputs = get_llava_inputs_outputs(img_np, model, processor, generate_config=generate_config)

    modified_token_ids, modified_token_list = ag.decode_tokens(inputs, outputs)
    print("Modified generated tokens: ", modified_token_list)

    matched_token_id_list = ag.get_selected_token_idx(modified_token_list, "▁horse")
    print("Modified generated tokens: ", matched_token_id_list)

    output_token_idx = ag.modified_token_idx_to_output_idx(matched_token_id_list[0])
    print(f"First selected token appears at {output_token_idx} of generated tokens")

    atten_weights = ag.get_attention_scores(outputs, token_idx=output_token_idx)
    print(f"Attention weights at {output_token_idx}: ", atten_weights.shape)

    agg_atten = ag.aggregate_attention(atten_weights, agg="avg")
    print(f"Average attention weights at {output_token_idx}: ", atten_weights.shape)

    text_atten, image_atten = ag.attention_maps(agg_atten, modified_token_ids)
    print("Attention of compressed text: ", text_atten.shape)

    vis = AttentionVisualizer(patch_size=14, image_size=336)
    for token_idx in matched_token_id_list:
        output_token_idx = ag.modified_token_idx_to_output_idx(token_idx)
        atten_weights = ag.get_attention_scores(outputs, token_idx=output_token_idx)
        agg_atten = ag.aggregate_attention(atten_weights, agg="avg")
        text_atten, _ = ag.attention_maps(agg_atten, modified_token_ids)
        vis.plot_text_attention(text_atten, modified_token_list, layer=-1, head=-1)
    
    

if __name__ == "__main__":
    test()