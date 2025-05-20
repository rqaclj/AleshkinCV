
import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

FORMATS = ["png", "jpg", "webp"]
INPUT_FOLDER = "data/output"
NUM_IMAGES = 5

class FormatEvaluator:
    def __init__(self, input_folder=INPUT_FOLDER, formats=FORMATS, num_images=NUM_IMAGES):
        self.input_folder = input_folder
        self.formats = formats
        self.num_images = num_images

    def evaluate(self):
        results = {fmt: {"SSIM": [], "MSE": []} for fmt in self.formats}
        for i in range(self.num_images):
            clean_path = os.path.join(self.input_folder, f"clean_image_{i}.png")
            clean_img = cv2.imread(clean_path)
            if clean_img is None:
                print(f"Не найдено: {clean_path}")
                continue

            for fmt in self.formats:
                encoded_path = os.path.join(self.input_folder, f"clean_image_{i}_encoded.{fmt}")
                if not os.path.exists(encoded_path):
                    cv2.imwrite(encoded_path, clean_img)

                encoded_img = cv2.imread(encoded_path)
                if encoded_img is None:
                    continue

                try:
                    ssim_score = ssim(clean_img, encoded_img, channel_axis=2, data_range=255)
                    mse_score = mean_squared_error(clean_img.flatten(), encoded_img.flatten())
                except Exception:
                    ssim_score = np.nan
                    mse_score = np.nan

                results[fmt]["SSIM"].append(ssim_score)
                results[fmt]["MSE"].append(mse_score)

        table_data = {
            fmt: {
                "SSIM": np.nanmean(results[fmt]["SSIM"]),
                "MSE": np.nanmean(results[fmt]["MSE"])
            } for fmt in self.formats
        }
        df = pd.DataFrame.from_dict(table_data, orient="index")
        print(df)

def main():
    evaluator = FormatEvaluator()
    evaluator.evaluate()
