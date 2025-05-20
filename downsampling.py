
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

INPUT_FOLDER = "data/output"
IMAGE_FORMAT = ".png"
DOWNSAMPLE_FACTOR = 2
NUM_IMAGES = 5

class ImageDownsampler:
    def __init__(self, folder=INPUT_FOLDER, fmt=IMAGE_FORMAT, factor=DOWNSAMPLE_FACTOR, count=NUM_IMAGES):
        self.image_folder = folder
        self.image_format = fmt
        self.downsample_factor = factor
        self.num_images = count
        self.ssim_max_values = []
        self.mse_max_values = []
        self.ssim_median_values = []
        self.mse_median_values = []

    def downsample_max(self, image):
        h, w, c = image.shape
        new_h = h // self.downsample_factor
        new_w = w // self.downsample_factor
        downsampled = np.zeros((new_h, new_w, c), dtype=image.dtype)
        for i in range(new_h):
            for j in range(new_w):
                block = image[i*self.downsample_factor:(i+1)*self.downsample_factor,
                              j*self.downsample_factor:(j+1)*self.downsample_factor]
                downsampled[i, j] = np.max(block, axis=(0, 1))
        return downsampled

    def downsample_median(self, image):
        h, w, c = image.shape
        new_h = h // self.downsample_factor
        new_w = w // self.downsample_factor
        downsampled = np.zeros((new_h, new_w, c), dtype=image.dtype)
        for i in range(new_h):
            for j in range(new_w):
                block = image[i*self.downsample_factor:(i+1)*self.downsample_factor,
                              j*self.downsample_factor:(j+1)*self.downsample_factor]
                downsampled[i, j] = np.median(block, axis=(0, 1))
        return downsampled.astype(np.uint8)

    def process_images(self):
        for i in range(self.num_images):
            path = os.path.join(self.image_folder, f'clean_image_{i}{self.image_format}')
            img = cv2.imread(path)
            if img is None:
                print(f"Ошибка загрузки: {path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            down_max = self.downsample_max(img)
            down_med = self.downsample_median(img)

            cv2.imwrite(os.path.join(self.image_folder, f"clean_image_{i}_downsampled_max{self.image_format}"),
                        cv2.cvtColor(down_max, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.image_folder, f"clean_image_{i}_downsampled_median{self.image_format}"),
                        cv2.cvtColor(down_med, cv2.COLOR_RGB2BGR))

            img_resized = cv2.resize(img, (down_max.shape[1], down_max.shape[0]), interpolation=cv2.INTER_LINEAR)
            self.ssim_max_values.append(ssim(img_resized, down_max, channel_axis=-1))
            self.mse_max_values.append(mean_squared_error(img_resized.flatten(), down_max.flatten()))
            self.ssim_median_values.append(ssim(img_resized, down_med, channel_axis=-1))
            self.mse_median_values.append(mean_squared_error(img_resized.flatten(), down_med.flatten()))

            self.display(img, down_max, down_med, i)

        self.print_metrics()

    def display(self, orig, max_d, med_d, idx):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(orig)
        plt.title(f"Оригинал {idx+1}")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(max_d)
        plt.title("Downsample Max")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(med_d)
        plt.title("Downsample Median")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def print_metrics(self):
        print("SSIM Max:", self.ssim_max_values)
        print("MSE Max:", self.mse_max_values)
        print("SSIM Median:", self.ssim_median_values)
        print("MSE Median:", self.mse_median_values)

def main():
    processor = ImageDownsampler()
    processor.process_images()
