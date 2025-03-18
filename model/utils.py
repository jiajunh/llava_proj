import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig


def load_llava(model_name_or_path,
               device_map="auto",
               padding_side="left",
               quantization = True):

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=quantization,
    #     bnb_4bit_compute_dtype=torch.float16
    # )

    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path,
                                                          device_map=device_map,
                                                          load_in_4bit = quantization,
                                                          torch_dtype=torch.float16)
    
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    return model, tokenizer, processor

