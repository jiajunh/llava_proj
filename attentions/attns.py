import torch

class AttentionGenerator:
    def __init__(self, model, processor, tokenizer,
                 prompt=None):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_token = "<image>"
        self.image_token_id = self.model.config.image_token_index
        self.image_size = self.model.config.vision_config.image_size
        self.patch_size = self.model.config.vision_config.patch_size
        self.image_token_num = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)
        self.prompt = "USER: <image>\n describe the image ASSISTANT:" if prompt is None else prompt
        self.prompt_token_length = 0

    def get_attention_scores(self, outputs, token_idx=-1):
        ## [new_tokens, layers, batch_size, num_head, input_size, input_size]
        atten_weight = []
        for layer in outputs["attentions"][token_idx]:
            layer_attns = layer.squeeze(0)
            if token_idx == 0:
                atten_weight.append(layer_attns.cpu())  
            else:
                atten_weight.append(layer_attns[:, -1, :].cpu())            
        return torch.stack(atten_weight)
    
    def aggregate_attention(self, atten_weights, agg="avg", layer_lists=None):
        # atten_weights, [layers, heads, pre_tokens]
        # agg: aggregation level
        #     "head":  no aggregation, return [layers, heads, pre_tokens]
        #     "avg": average over each layers return [layers, 1, pre_tokens]
        if agg == "head":
            return atten_weights
        if agg == "avg":
            return atten_weights.mean(dim=1, keepdims=True)
        return atten_weights[layer_lists].mean(dim=1, keepdims=True).mean(dim=0, keepdims=True)
    
    def get_image_atten_for_every_new_token(self, outputs, modified_token_ids, agg="avg"):
        # modified_token_ids: compress 567 image tokens to 1 
        num_token = len(outputs["attentions"])
        img_token_idx = modified_token_ids.index(self.image_token_id)
        atten_score = []
        for i in range(num_token):
            atten_weights = self.get_attention_scores(outputs, i)
            if i == 0:
                atten_weights = atten_weights[:,:,-1,:]
            agg_atten = self.aggregate_attention(atten_weights, agg=agg)
            atten_score.append(torch.sum(agg_atten[:,:,img_token_idx:img_token_idx+self.image_token_num], dim=-1)[-1,:].squeeze().cpu().item())
        return atten_score
    

    def decode_tokens(self, inputs, outputs):
        input_ids = inputs["input_ids"].cpu()
        output_ids = outputs["sequences"].cpu()

        prompt_length = input_ids.shape[1]
        self.prompt_token_length = prompt_length
    
        output_ids_list = output_ids.squeeze().tolist()
        img_token_first_idx = output_ids_list.index(self.image_token_id)
    
        modified_output_token_ids = output_ids_list[0:img_token_first_idx] + [self.image_token_id] + output_ids_list[img_token_first_idx+self.image_token_num:]
        decode_tokens = self.tokenizer.convert_ids_to_tokens(modified_output_token_ids)
        return modified_output_token_ids, decode_tokens
    

    def get_selected_token_idx(self, modified_token_list, input_token):
        matched_idx = [i for i, token in enumerate(modified_token_list) if token==input_token]
        return matched_idx
    
    def modified_token_idx_to_output_idx(self, modified_token_idx):
        return modified_token_idx + self.image_token_num - 1 - self.prompt_token_length
    
    def attention_maps(self, agg_atten, modified_token_ids):
        img_token_idx = modified_token_ids.index(self.image_token_id)

        text_atten = torch.cat([
            agg_atten[:,:,0:img_token_idx],
            torch.sum(agg_atten[:,:,img_token_idx:img_token_idx+self.image_token_num], dim=-1, keepdims=True),
            agg_atten[:,:,img_token_idx+self.image_token_num:]
        ], dim=-1)

        image_atten = agg_atten[:,:,img_token_idx:img_token_idx+self.image_token_num]
        return text_atten, image_atten
    
    def prompt_patch_attention(self, outputs, modified_token_list, patch_idx, 
                               select_layer=-1, select_head=-1, prompt_agg=True):
        image_start_idx = modified_token_list.index(self.image_token)
        prompt_agg_atten = self.get_attention_scores(outputs, token_idx=0)
        if prompt_agg:
            prompt_agg_atten = prompt_agg_atten.mean(dim=1, keepdims=True)
        selected_prompt_agg_atten = prompt_agg_atten[select_layer][select_head][image_start_idx+patch_idx-1]
        return selected_prompt_agg_atten[image_start_idx:image_start_idx+self.image_token_num].reshape((1,1,-1)).cpu()
    

    def sort_patch_index_for_token(self, image_atten,
                                   select_layer=-1, select_head=-1):
        # output_token_idx = self.modified_token_idx_to_output_idx(token_idx)
        # atten_weights = self.get_attention_scores(outputs, token_idx=output_token_idx)
        # if prompt_agg:
        #     agg_atten = self.aggregate_attention(atten_weights, agg="avg")
        # else:
        #     agg_atten = self.aggregate_attention(atten_weights, agg="head")
        # _, image_atten = self.attention_maps(agg_atten, modified_token_ids)

        image_atten_for_token = image_atten[select_layer, select_head].reshape((1,1,-1))
        image_atten_for_token_prev_layer = image_atten[select_layer-1, select_head].reshape((1,1,-1))
        
        _, sorted_indices = torch.sort(image_atten_for_token, dim=2, descending=True)
        sorted_indices = sorted_indices.squeeze().numpy()
        return sorted_indices, image_atten_for_token, image_atten_for_token_prev_layer


