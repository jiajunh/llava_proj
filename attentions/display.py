import cv2
import torch 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class AttentionVisualizer:
    def __init__(self, patch_size=14, image_size=336):
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_row = self.image_size // self.patch_size
        self.n_col = self.image_size // self.patch_size
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
    
    def generate_heatmap(self, atten_map, fancy=False, mul=1.2):
        cam = atten_map
        if not fancy:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            heads, height, width = atten_map.shape
            cam = cam.reshape((heads, -1))
            cam_min = np.min(cam, axis=1, keepdims=True)
            cam_max = np.max(cam, axis=1, keepdims=True)
            cam = (cam - cam_min) / (cam_max - cam_min)
            cam = cam.reshape(atten_map.shape) * mul
        cam_img = np.uint8(255 * cam)    
        heatmap = [cv2.applyColorMap(cam_img_head, cv2.COLORMAP_HSV) for cam_img_head in cam_img]
        heatmap = np.array(heatmap)
        heatmap = np.float32(heatmap) / 255
        return heatmap
    
    def interpolate_attention_map(self, image_atten, img_h, img_w):
        layers, heads, tokens = image_atten.shape
        atten_map = image_atten.reshape((layers, heads, self.n_row, self.n_col)).cpu()
        atten_map = torch.nn.functional.interpolate(atten_map, size=(img_h, img_w), mode="bilinear").numpy()
        return atten_map
    
    def get_mixed_images(self, image_atten, image, avg=False, fancy=False):
        img_np = np.asarray(image)
        height, width, _ = img_np.shape
        img_np = self.resize_image(img_np, (width // self.n_col * self.n_col, height // self.n_row * self.n_row))
        img = np.float32(img_np) / 255
        height, width, _ = img_np.shape

        atten_map = self.interpolate_attention_map(image_atten, height, width)

        if avg is True:
            avg_attn = atten_map.mean(axis=1, keepdims=True)
            heatmap = np.array([self.generate_heatmap(avg_layer, fancy) for avg_layer in avg_attn])
        else:
            heatmap = np.array([self.generate_heatmap(avg_layer, fancy) for avg_layer in atten_map])
        mixed_imgs = img * 0.5 + heatmap * 0.4
        return mixed_imgs
    

    def plot_image_atten(self, image_atten, image, plot_layers=[],
                         avg=False, fancy=False):
        
        mixed_imgs = self.get_mixed_images(image_atten, image, avg, fancy)

        if avg:
            fig, ax = plt.subplots(8, 4, figsize=(30, 40))
            for i in range(32):
                row = i // 4
                col = i % 4
                ax[row,col].imshow(mixed_imgs[i,0])
                ax[row,col].set_axis_off()
                ax[row,col].set_title("Average of Layer {}".format(i), fontsize=16)

        else:
            if len(plot_layers) == 0:
                plot_layers = [-1]
            fig, ax = plt.subplots(8*len(plot_layers), 4, figsize=(30, 40*len(plot_layers)))
            for k, layer in enumerate(plot_layers):
                for i in range(32):
                    row = i // 4
                    col = i % 4
                    ax[k*8+row,col].imshow(mixed_imgs[layer, i])
                    ax[k*8+row,col].set_axis_off()
                    ax[k*8+row,col].set_title("Layer {}, head {}".format(layer, i), fontsize=16)
        return fig

    


        # img_np = np.asarray(image)
        # height, width, _ = img_np.shape
        # img_np = self.resize_image(img_np, (width // self.n_col * self.n_col, height // self.n_row * self.n_row))
        # img = np.float32(img_np) / 255
        # height, width, _ = img_np.shape

        # atten_map = self.interpolate_attention_map(image_atten, height, width)

        # if avg is True:
        #     avg_attn = atten_map.mean(axis=1, keepdims=True)

        #     fig, ax = plt.subplots(8,4, figsize=(30, 40))
        #     # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.05)
        #     for i in range(32):
        #         row = i // 4
        #         col = i % 4

        #         heatmap = self.generate_heatmap(avg_attn[i], fancy)
        #         attn_plot = img * 0.5 + heatmap * 0.4
                
        #         ax[row,col].imshow(attn_plot[0])
        #         ax[row,col].set_axis_off()
        #         ax[row,col].set_title("Average of Layer {}".format(i), fontsize=20)

        # else:
        #     for layer in plot_layers:
        #         if len(plot_heads) == 0:
        #             fig, ax = plt.subplots(8, 4, figsize=(30, 40))
        #             fig.suptitle("attention map of heads in layer {}".format(layer), fontsize=30, x=0.5, y=0.92, horizontalalignment="center")

        #             heatmap = self.generate_heatmap(atten_map[layer], fancy)
                    
        #             for i in range(32):
        #                 row = i // 4
        #                 col = i % 4

        #                 attn_plot = img * 0.5 + heatmap[i] * 0.4

        #                 ax[row,col].imshow(attn_plot)
        #                 ax[row,col].set_axis_off()
        #                 ax[row,col].set_title("Layer {}, head {}".format(layer, i), fontsize=20)


    