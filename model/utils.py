import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig


def load_llava(model_name_or_path,
               device_map="auto",
               padding_side="left",
               quantization = True):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path,
                                                          device_map=device_map,
                                                          quantization_config=quantization_config)

    # model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path,
    #                                                       device_map=device_map)
    
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    return model, tokenizer, processor


def get_llava_image_features(model, processor, image):
    inputs = processor(images=image, text="", return_tensors="pt").to(model.device, torch.float16)
    image_features = model.get_image_features(
        pixel_values=inputs.pixel_values,
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy
    )
    return image_features

def get_llava_logits(image_features = None,
                     model=None,
                     tokenizer = None,
                     image_token = None):
        
        batch, _, _ = image_features.shape
        prompt_strings = [image_token]*batch
        text_inputs = tokenizer(prompt_strings, return_tensors="pt")
        input_ids = text_inputs["input_ids"]
        
        inputs_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds, _, _, _ = model._merge_input_ids_with_image_features(image_features, 
                                                                            inputs_embeds, 
                                                                            input_ids, 
                                                                            text_inputs.attention_mask, 
                                                                            labels=None)
        outputs = model.language_model(
            # attention_mask=attention_mask,
            # position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        logits = outputs[0]
        return logits

def get_llava_inputs_outputs(image, model, processor, prompt=None, generate_config=None):
    prompt = "USER: <image>\n describe the image ASSISTANT:" if prompt is None else prompt
    if generate_config is None:
        generate_config = {
            "return_dict_in_generate": True,
            "output_attentions": True,
            "output_hidden_states": True,
        }
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    input_kwargs = {**inputs, **generate_config}
    outputs = model.generate(**input_kwargs)
    return inputs, outputs
