import torch 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class AttentionVisualizer:
    def __init__(self, patch_size=14, image_size=336):
        self.patch_size = patch_size
        self.image_size = image_size
        self.dpi = 100
        # self.horizontal_space = 3 # pixels
        # self.vertical_space = 3

    def resize_image(self, image, resized_size):
        pil_image = Image.fromarray(image)
        resized_img = pil_image.resize(resized_size, Image.BICUBIC)
        image = np.asarray(resized_img)
        return image
    
    def plot_text_attention(self, text_atten, modified_token_list, layer=-1, head=-1):
        fig, ax = plt.subplots(2,1, figsize=(10, 5))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.8)

        plot_attention = text_atten[layer, head, :].cpu()
        plot_decode_tokens = modified_token_list[:text_atten.shape[-1]]
        
        ax[0].scatter(range(len(plot_decode_tokens)),plot_attention, linewidth=3, marker='x', s=80, color='r')
        ax[0].tick_params(axis='x', labelsize=12)
        ax[0].tick_params(axis='y', labelsize=12)
        ax[0].set_xticks(range(len(plot_decode_tokens)))
        ax[0].set_xticklabels(plot_decode_tokens, rotation=75)
        ax[0].set_title("Sum vision tokens with first token", fontsize=16)
        ax[0].grid()
        
        ax[1].scatter(range(len(plot_decode_tokens)-1),plot_attention[1:], linewidth=3, marker='x', s=80, color='r')
        ax[1].tick_params(axis='x', labelsize=12)
        ax[1].tick_params(axis='y', labelsize=12)
        ax[1].set_xticks(range(len(plot_decode_tokens)-1))
        ax[1].set_xticklabels(plot_decode_tokens[1:], rotation=75)
        ax[1].set_title("Sum vision tokens without first token", fontsize=16)
        ax[1].grid()
        return fig
    
    def plot_image_atten_for_each_token(self, atten_score, modified_token_list):
        plot_decode_tokens = modified_token_list[-len(atten_score):]
        fig, ax = plt.subplots(1, figsize=(12, 4))
        ax.scatter(range(len(atten_score)),atten_score, linewidth=2, marker='x', s=60, color='r')
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xticks(range(len(plot_decode_tokens)))
        ax.set_xticklabels(plot_decode_tokens, rotation=75)
        ax.set_title("image attention weights for each new token", fontsize=14)
        ax.grid()
        return fig
    