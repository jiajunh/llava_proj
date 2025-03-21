import torch
import nltk

from model.utils import get_llava_image_features, get_llava_logits


class LogitLens:
    def __init__(self, model, processor, tokenizer, 
                 prompt=None):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_token = "<image>"
        self.image_token_id = self.model.config.image_token_index
        self.image_size = self.model.config.vision_config.image_size
        self.patch_size = self.model.config.vision_config.patch_size
        self.prompt = "USER: <image>\n describe the image ASSISTANT:" if prompt is None else prompt


    def get_topk_logits(self, logits, topk=5):
        topk_values, topk_indices = torch.topk(logits, topk, dim=-1)
        return topk_values, topk_indices
    

    def normalize_tokens(self, token_list):
        normalized_token_list = []
        for token in token_list:
            if len(token)>1 and token[0] == '▁':
                normalized_token_list.append(token[1:])
            else:
                normalized_token_list.append(token)
        return normalized_token_list
    
    def get_generated_ids(self, image, topk=20):
        image_features = get_llava_image_features(self.model, self.processor, image)
        image_features = torch.permute(image_features, (1, 0, 2))
    
        logits = get_llava_logits(image_features,
                                  model=self.model,
                                  tokenizer = self.tokenizer,
                                  image_token = self.image_token)
        
        _, topk_indices = self.get_topk_logits(logits, topk)
        generate_ids = topk_indices[:,-1,:]
        return generate_ids
    
    def batch_generate(self, image, topk=5):
        generate_ids = self.get_generated_ids(image, topk=topk)
        id_list = generate_ids.tolist()

        next_tokens = [self.normalize_tokens(self.tokenizer.convert_ids_to_tokens(id)) for id in id_list]
        return next_tokens
    
    def get_most_frequent_token_ids(self, generate_ids, k=30):
        unique_values, counts = torch.unique(generate_ids, return_counts=True)
        _, indices = torch.sort(counts, descending=True)
        sorted_unique_values = unique_values[indices]
        return sorted_unique_values[0:k]
    
    def decode(self, generate_ids):
        id_list = generate_ids.tolist()
        tokens = self.normalize_tokens(self.tokenizer.convert_ids_to_tokens(id_list))
        return tokens
    
    def filter_tokens(self, tokens):
        tag_set = {"NN", "NNS", "NNP"}
        normalized_tokens = self.normalize_tokens(tokens)
        tagged_tokens = nltk.pos_tag(normalized_tokens)
        filtered_tokens = []    
        for (tk, tag) in tagged_tokens:
            if len(tk) > 2 and tk.isalnum() and tag in tag_set:
                filtered_tokens.append(tk)
        return filtered_tokens
    
    def get_token_id(self, token):
        ids = self.tokenizer.convert_tokens_to_ids(token)
        return ids

    def get_patch_mask(self, generated_ids, given_id):
        mask = torch.where(torch.isin(generated_ids, given_id), 1, 0)
        return mask

    def patch_with_given_token(self, image, input_token, topk=20):
        # need to expand token to a list, eg: sign -> [sign, signs, ▁sign, ...]
        expanded_tokens = [input_token]
        if input_token[0] == "▁" and len(input_token) > 1:
            expanded_tokens.append(input_token[1:])
        if input_token[0] != "▁":
            expanded_tokens.append("▁" + input_token)
        print("!!!!!!", input_token, expanded_tokens)

        generate_ids = self.get_generated_ids(image, k=topk)
        mask = torch.zeros(generate_ids.shape)
        for token in expanded_tokens:
            input_token_id = self.get_token_id(token) 
            mask1 = self.get_patch_mask(generate_ids, input_token_id)
            mask = torch.logical_or(mask, mask1).int()
        return mask.sum(dim=1)