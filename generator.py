
import cv2
import numpy as np
import os
import random
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import time
import matplotlib.pyplot as plt

# Конфигурационные параметры
DATASET_PATH = "data"
OUTPUT_DIR = "data/output"
TILE_SIZE = (64, 64)
CANVAS_SIZE = (512, 512)
CELL_COUNT = (3, 15)
NUM_IMAGES = 5  # Сколько изображений генерировать
BACKGROUND_RANGE = (180, 220)

Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

class BloodCellGenerator:
    def __init__(self):
        self.cell_templates = self._load_images("patch")[:5]
        self.background_templates = self._load_images("fon")[:5]

        if not self.cell_templates or not self.background_templates:
            raise ValueError("Не найдены патчи или фоны в директории 'data/patch' и 'data/fon'")

    def _load_images(self, subfolder):
        files = list((Path(DATASET_PATH) / subfolder).glob("*.png"))
        return [cv2.imread(str(f)) for f in files if f.is_file()]

    def _generate_artificial_cell(self):
        size = random.randint(TILE_SIZE[0]//2, TILE_SIZE[0])
        cell = np.zeros((size, size, 3), dtype=np.uint8)
        color = tuple(random.randint(50, 200) for _ in range(3))
        cv2.circle(cell, (size//2, size//2), size//2, color, -1)
        return cv2.GaussianBlur(cell, (5,5), 0)

    def _create_composite_background(self):
        if random.random() < 0.5:
            bg_color = random.randint(*BACKGROUND_RANGE)
            background = np.full((*CANVAS_SIZE[::-1], 3), bg_color, dtype=np.uint8)
        else:
            background = self._generate_tiled_background()
        return cv2.medianBlur(background, 3)

    def _generate_tiled_background(self):
        canvas = np.zeros((*CANVAS_SIZE[::-1], 3), dtype=np.uint8)
        tiles_x = CANVAS_SIZE[0] // TILE_SIZE[0]
        tiles_y = CANVAS_SIZE[1] // TILE_SIZE[1]

        for y in range(tiles_y):
            for x in range(tiles_x):
                tile = random.choice(self.background_templates)
                tile = cv2.resize(tile, TILE_SIZE)
                y1, y2 = y*TILE_SIZE[1], (y+1)*TILE_SIZE[1]
                x1, x2 = x*TILE_SIZE[0], (x+1)*TILE_SIZE[0]
                canvas[y1:y2, x1:x2] = tile
        return canvas

    def _blend_cells(self, background):
        composite = background.copy()
        num_cells = random.randint(*CELL_COUNT)

        for _ in range(num_cells):
            if random.random() < 0.7:
                cell = random.choice(self.cell_templates)
            else:
                cell = self._generate_artificial_cell()

            cell = cv2.resize(cell, (random.randint(40,80),)*2)
            mask = 255 * np.ones(cell.shape[:2], dtype=np.uint8)
            pos = (random.randint(0, CANVAS_SIZE[0]-cell.shape[1]),
                   random.randint(0, CANVAS_SIZE[1]-cell.shape[0]))
            center = (pos[0] + cell.shape[1]//2, pos[1] + cell.shape[0]//2)

            composite = cv2.seamlessClone(cell, composite, mask, center, cv2.NORMAL_CLONE)
        return composite

    def generate_sample(self):
        base = self._create_composite_background()
        clean_image = self._blend_cells(base)
        return clean_image

def main():
    generator = BloodCellGenerator()
    for i in range(NUM_IMAGES):
        clean = generator.generate_sample()
        cv2.imwrite(str(Path(OUTPUT_DIR) / f"clean_image_{i}.png"), clean)
