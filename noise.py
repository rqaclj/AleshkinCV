
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time
from pathlib import Path

INPUT_DIR = "data/output"
OUTPUT_DIR = "data/output"
NUM_IMAGES = 5

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

class NoiseProcessor:
    @staticmethod
    def apply_gaussian_noise(img, intensity=15):
        noise = np.random.normal(0, intensity, img.shape).astype(np.int16)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_uniform_noise(img, strength=25):
        return cv2.add(img, np.uint8(strength * np.random.rand(*img.shape)))

class RestorationMetrics:
    @staticmethod
    def calculate_mse(orig, processed):
        return np.mean((orig - processed) ** 2)

    @staticmethod
    def calculate_ssim(orig, processed):
        return ssim(orig, processed, channel_axis=2, data_range=255)

def benchmark_denoising(noisy_img, clean_img):
    results = []

    # Median
    start = time.perf_counter()
    denoised = cv2.medianBlur(noisy_img, 5)
    results.append(("Median", time.perf_counter() - start,
                    RestorationMetrics.calculate_mse(clean_img, denoised),
                    RestorationMetrics.calculate_ssim(clean_img, denoised)))

    # Gaussian
    start = time.perf_counter()
    denoised = cv2.GaussianBlur(noisy_img, (9,9), 2)
    results.append(("Gaussian", time.perf_counter() - start,
                    RestorationMetrics.calculate_mse(clean_img, denoised),
                    RestorationMetrics.calculate_ssim(clean_img, denoised)))

    # Bilateral
    start = time.perf_counter()
    denoised = cv2.bilateralFilter(noisy_img, 15, 60, 60)
    results.append(("Bilateral", time.perf_counter() - start,
                    RestorationMetrics.calculate_mse(clean_img, denoised),
                    RestorationMetrics.calculate_ssim(clean_img, denoised)))

    # Non-local Means
    start = time.perf_counter()
    denoised = cv2.fastNlMeansDenoisingColored(noisy_img, None, 10, 10, 7, 21)
    results.append(("NLM", time.perf_counter() - start,
                    RestorationMetrics.calculate_mse(clean_img, denoised),
                    RestorationMetrics.calculate_ssim(clean_img, denoised)))

    return results

def visualize_results(images, metrics):
    plt.figure(figsize=(18, 12))
    titles = [
        'Original Image', 
        'Gaussian Noise', 
        'Uniform Noise',
        'Median', 
        'Gaussian',
        'Bilateral', 
        'NLM'
    ]

    for idx, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(3, 3, idx+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("\nMetrics:")
    print(f"{'Method':<10} | {'Time':<8} | {'MSE':<8} | {'SSIM'}")
    for name, t, m, s in metrics:
        print(f"{name:<10} | {t:.4f} | {m:.2f} | {s:.4f}")

def main():
    processor = NoiseProcessor()
    for i in range(NUM_IMAGES):
        clean_path = Path(INPUT_DIR) / f"clean_image_{i}.png"
        clean = cv2.imread(str(clean_path))

        gaussian_noise = processor.apply_gaussian_noise(clean)
        uniform_noise = processor.apply_uniform_noise(clean)

        gauss_metrics = benchmark_denoising(gaussian_noise, clean)

        denoised_images = [
            clean,
            gaussian_noise,
            uniform_noise,
            cv2.medianBlur(gaussian_noise, 5),
            cv2.GaussianBlur(gaussian_noise, (9,9), 2),
            cv2.bilateralFilter(gaussian_noise, 15, 60, 60),
            cv2.fastNlMeansDenoisingColored(gaussian_noise, None, 10, 10, 7, 21)
        ]

        visualize_results(denoised_images, gauss_metrics)

        # Сохраняем
        cv2.imwrite(str(Path(OUTPUT_DIR) / f"gauss_noise_{i}.png"), gaussian_noise)
        cv2.imwrite(str(Path(OUTPUT_DIR) / f"uniform_noise_{i}.png"), uniform_noise)
