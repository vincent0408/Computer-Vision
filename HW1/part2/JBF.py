import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        pad_w, wndw_size = self.pad_w, self.wndw_size
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, pad_w, pad_w, pad_w, pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, pad_w, pad_w, pad_w, pad_w, BORDER_TYPE).astype(np.int32)
        output, grid_length, isRGB = np.empty(img.shape), np.arange(-pad_w, pad_w + 1), len(padded_guidance.shape) == 3
        x_s, y_s = np.meshgrid(grid_length, grid_length)        
        spatial_kernel = -0.5 * (x_s ** 2 + y_s ** 2) * self.sigma_s ** -2
        lut = np.arange(256) ** 2 * (255 * self.sigma_r) ** -2 * -0.5

        # Method 1: 2d window

        # padded_guidance = padded_guidance / 255
        # h, w = padded_guidance.shape[0], padded_guidance.shape[1]

        # for x in range(pad_w, w - pad_w):
        #     x_l, x_h = x - pad_w, x + pad_w + 1
        #     for y in range(pad_w, h - pad_w):
        #         y_l, y_h = y - pad_w, y + pad_w + 1
        #         range_kernel = (padded_guidance[y_l : y_h, x_l : x_h] - \
        #                         padded_guidance[y, x]) ** 2 * self.sigma_r ** -2 * -0.5
        #         if(isRGB):
        #             range_kernel = np.sum(range_kernel, axis=2)
        #         w_matrix = np.exp(spatial_kernel + range_kernel)
        #         for rgb in [0, 1, 2]:
        #             output[y - pad_w, x - pad_w, rgb] = np.sum(padded_img[y_l : y_h, x_l : x_h, rgb] * w_matrix) / np.sum(w_matrix)
        
        # Method 2: naive way to speed up
        # Spereate single channel and color images before loop starts
        
        # Method 3: range kernel lookup table + range window computation reduced + Method 2

        top_row_window = padded_guidance[:wndw_size, :wndw_size]
        if(isRGB):
            h, w, _ = padded_guidance.shape
            x_end, y_end, x_l, x_h = w - pad_w - 1, h - pad_w - 1, 0, wndw_size
            for x in range(pad_w, w - pad_w):
                y_l, y_h, window = 0, wndw_size, top_row_window 
                for y in range(pad_w, h - pad_w):
                    range_kernel = np.sum(lut[np.absolute(window - padded_guidance[y, x])], axis=2)
                    w_matrix = np.exp(spatial_kernel + range_kernel)
                    output[y_l, x_l] = np.einsum('ijk,ij->k', padded_img[y_l : y_h, x_l : x_h], w_matrix) / np.sum(w_matrix)
                    if(y == y_end):
                        break
                    window = np.concatenate((window[1:], padded_guidance[y_h, x_l:x_h].reshape(1, wndw_size, 3)), axis=0)
                    y_l, y_h = y_l + 1, y_h + 1
                if(x == x_end):
                    break
                top_row_window = np.concatenate((top_row_window[:, 1:], padded_guidance[:wndw_size, x_h].reshape(wndw_size, 1, 3)), axis=1)
                x_l, x_h = x_l + 1, x_h + 1
        else:
            h, w = padded_guidance.shape
            x_end, y_end, x_l, x_h = w - pad_w - 1, h - pad_w - 1, 0, wndw_size

            for x in range(pad_w, w - pad_w):
                y_l, y_h, window = 0, wndw_size, top_row_window 

                for y in range(pad_w, h - pad_w):
                    range_kernel = lut[np.absolute(window - padded_guidance[y, x])]
                    w_matrix = np.exp(spatial_kernel + range_kernel)
                    output[y_l, x_l] = np.einsum('ijk,ij->k', padded_img[y_l : y_h, x_l : x_h], w_matrix) / np.sum(w_matrix)
                    if(y == y_end):
                        break
                    window = np.concatenate((window[1:], padded_guidance[y_h, x_l:x_h].reshape(1, wndw_size)))
                    y_l, y_h = y_l + 1, y_h + 1
                if(x == x_end):
                    break
                top_row_window = np.concatenate((top_row_window[:, 1:], padded_guidance[:wndw_size, x_h].reshape(wndw_size, 1)), axis=1)
                x_l, x_h = x_l + 1, x_h + 1
        return np.clip(output, 0, 255).astype(np.uint8)