import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class LogitLensVisualizer:
    def __init__(self, patch_size=14, image_size=336):
        self.patch_size = patch_size
        self.image_size = image_size
        self.dpi = 100
        self.horizontal_space = 2 # pixels
        self.vertical_space = 2

    def resize_image(self, image, resized_size):
        pil_image = Image.fromarray(image)
        resized_img = pil_image.resize(resized_size, Image.BICUBIC)
        image = np.asarray(resized_img)
        return image

    def plot_tokens_on_image(self, image, tokens, 
                             show_full_image=False, 
                             part_idx=0,
                             n_splits=3,
                             use_resized_img=False,
                             text_color="yellow",
                             text_fontsize=5):
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        
        height, width, _ = image.shape

        if show_full_image:
            n_splits = 1

        if use_resized_img:
            image = self.resize_image(image, (self.image_size, self.image_size))
        else:
            image = self.resize_image(image, (width//self.patch_size*self.patch_size, height//self.patch_size*self.patch_size))

        height, width, _ = image.shape
        
        nrow = self.image_size // self.patch_size
        ncol = self.image_size // self.patch_size

        step_h = height // nrow
        step_w = width // ncol

        part_row = part_idx // n_splits
        part_col = part_idx % n_splits

        width_in = (width + (ncol + 1) * self.vertical_space) / self.dpi
        height_in = (height + (nrow + 1) * self.horizontal_space) / self.dpi

        plt.subplots_adjust(left=1.0 * self.vertical_space / self.dpi,
                            right=1 - 1.0 * self.vertical_space / self.dpi, 
                            top=1 - 1.0 * self.horizontal_space / self.dpi, 
                            bottom=1.0 * self.horizontal_space / self.dpi, 
                            wspace=1.0 * self.vertical_space / self.dpi, 
                            hspace=1.0 * self.horizontal_space / self.dpi)
        
        fig, ax = plt.subplots(nrow//n_splits, nrow//n_splits, figsize=(width_in, height_in))
        for i in range(nrow // n_splits):
            for j in range(ncol // n_splits):
                patch_row = part_row * (nrow // n_splits) + i
                patch_col = part_col * (ncol // n_splits) + j
                ax[i,j].imshow(image[patch_row*step_h:(patch_row+1)*step_h, patch_col*step_w:(patch_col+1)*step_w,:])
                ax[i,j].set_axis_off()

                feature_idx = patch_row * ncol + patch_col
                for k,s in enumerate(tokens[feature_idx]): 
                    ax[i,j].text(0.1*(width//ncol), (1+k)*((height//ncol)/(len(tokens[feature_idx])+1)), s, 
                                 fontsize=text_fontsize, 
                                 color=text_color, 
                                 weight='bold',
                                 )
        plt.savefig(f"./plots/tokens_on_image_{part_idx}_{n_splits}.png")
        print("*"*5, f"./plots/tokens_on_image_{part_idx}_{n_splits}.png saved!")
        return fig


    def plot_saliency_map(self, image, mask):

        img_np = np.asarray(image)
        height, width, _ = img_np.shape
        img_np = self.resize_image(img_np, (width // 24 * 24, height // 24 * 24))
        img = np.float32(img_np) / 255
        height, width, _ = img_np.shape

        mask = mask.reshape((1, 1, 24, 24)).float()
        mask = torch.nn.functional.interpolate(mask, size=(height, width), mode="nearest").numpy()
        
        atten_map = mask[0, 0]
        cam = atten_map
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        print(np.min(cam), np.max(cam), np.mean(cam))
        cam_img = np.uint8(255 * cam)
        print(np.min(cam_img), np.max(cam_img), np.mean(cam_img))
        heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_HSV)
        heatmap = np.float32(heatmap) / 255

        print(np.min(heatmap[0,:,:,0]), np.max(heatmap[0,:,:,0]), np.mean(heatmap[0,:,:,0]))
        print(np.min(heatmap[0,:,:,1]), np.max(heatmap[0,:,:,1]), np.mean(heatmap[0,:,:,1]))
        print(np.min(heatmap[0,:,:,2]), np.max(heatmap[0,:,:,2]), np.mean(heatmap[0,:,:,2]))

        # mul = 1.0
        # cam = atten_map
        # heads, height, width = atten_map.shape
        # cam = cam.reshape((heads, -1))
        # cam_min = np.min(cam, axis=1, keepdims=True)
        # cam_max = np.max(cam, axis=1, keepdims=True)
        # cam = (cam - cam_min) / (cam_max - cam_min)
        # cam = cam.reshape(atten_map.shape) * mul


        mixed_img = img * 0.5 + heatmap[0] * 0.5

        fig, ax = plt.subplots(1)
        ax.imshow(mixed_img)
        ax.set_axis_off()



        # height, width, _ = image.shape
        # n_row = self.image_size // self.patch_size
        # n_col = self.image_size // self.patch_size
        # image = self.resize_image(image, (width//self.patch_size*self.patch_size, height//self.patch_size*self.patch_size))
        # height, width, _ = image.shape

        # mask = mask.cpu().reshape((1, 1, n_row, n_col)).float()
        # print(torch.mean(mask))
        # mask = torch.nn.functional.interpolate(mask, size=(height, width), mode="nearest").numpy()
        # print(np.mean(mask))

        # atten_map = mask[0]
        # mul = 1.0
        # cam = atten_map
        # heads, height, width = atten_map.shape
        # cam = cam.reshape((heads, -1))
        # cam_min = np.min(cam, axis=1, keepdims=True)
        # cam_max = np.max(cam, axis=1, keepdims=True)
        # cam = (cam - cam_min) / (cam_max - cam_min)
        # cam = cam.reshape(atten_map.shape) * mul

        # cam_img = np.uint8(255 * cam)    
        # heatmap = [cv2.applyColorMap(cam_img_head, cv2.COLORMAP_HSV) for cam_img_head in cam_img]
        # heatmap = np.array(heatmap)
        # heatmap = np.float32(heatmap)

        # print(np.max(heatmap), np.min(heatmap), np.mean(heatmap), np.max(image), np.min(image), np.mean(image))
        # print(type(heatmap), type(image))

        # mixed_img = np.uint8(image * 0.5 + heatmap[0] * 0.5)



        # mask = mask.cpu().numpy().reshape((n_row, n_col)).astype(float)
        # seg_resized = (np.array(Image.fromarray(mask).resize((width, height), Image.BILINEAR)))

        # mask2 = np.repeat(mask, height // n_row , axis=0)
        # mask2 = np.repeat(mask2, width // n_col, axis=1)
        
        # fig, ax = plt.subplots(1)
        # ax.imshow(image, alpha=1.0)
        # colors = ["white", "yellow"]
        # cmap = plt.cm.colors.ListedColormap(colors)
        # ax.imshow(mask2*255, cmap=cmap, alpha=0.6)
        # ax.set_axis_off()

        # plt.savefig(f"./plots/logit_lens_saliency_map.png")
        # print("*"*5, f"./plots/logit_lens_saliency_map.png saved!")

        return fig