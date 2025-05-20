
import os
import random
import cv2
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

class CellAnimationGenerator:
    def __init__(self):
        self.background_patches = []
        self.cell_patches = []

    def load_assets(self, bg_folder, cell_folder):
        def load_patches(folder):
            images = []
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(folder, f))
                    if img is not None:
                        images.append(img)
            return images

        self.background_patches = load_patches(bg_folder)
        self.cell_patches = load_patches(cell_folder)

        if not self.background_patches:
            raise ValueError("No background images found!")
        if not self.cell_patches:
            raise ValueError("No cell images found!")

    def generate_background(self, output_size=(450, 450)):
        h, w = output_size
        base_patch = random.choice(self.background_patches)
        base_patch = cv2.resize(base_patch, (w, h))
        background = base_patch.copy()

        for _ in range(random.randint(6, 10)):
            patch = random.choice(self.background_patches)
            patch_size = random.randint(min(h,w)//3, min(h,w)//2)
            patch = cv2.resize(patch, (patch_size, patch_size))
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((patch_size//2, patch_size//2), angle, 1)
            patch = cv2.warpAffine(patch, M, (patch_size, patch_size))
            patch = cv2.convertScaleAbs(patch, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))

            x = random.randint(-patch_size//3, w - patch_size//3*2)
            y = random.randint(-patch_size//3, h - patch_size//3*2)

            mask = np.ones((patch_size, patch_size, 3), dtype=np.float32)
            cv2.circle(mask, (patch_size//2, patch_size//2), patch_size//2, (1,1,1), -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            mask *= random.uniform(0.3, 0.7)

            try:
                y1, y2 = max(y,0), min(y+patch_size,h)
                x1, x2 = max(x,0), min(x+patch_size,w)
                patch_y1 = max(-y, 0)
                patch_y2 = patch_y1 + (y2 - y1)
                patch_x1 = max(-x, 0)
                patch_x2 = patch_x1 + (x2 - x1)
                roi = mask[patch_y1:patch_y2, patch_x1:patch_x2]
                background[y1:y2, x1:x2] = background[y1:y2, x1:x2] * (1 - roi) + patch[patch_y1:patch_y2, patch_x1:patch_x2] * roi
            except:
                continue

        return cv2.GaussianBlur(background, (15, 15), 0)

    def overlay_cell(self, bg, cell, position):
        x_center, y_center = position
        h, w = cell.shape[:2]
        x, y = int(x_center - w/2), int(y_center - h/2)

        if x < 0 or y < 0 or x+w > bg.shape[1] or y+h > bg.shape[0]:
            return bg

        mask = np.zeros((h, w, 3), dtype=np.float32)
        cv2.circle(mask, (w//2, h//2), min(w,h)//2, (1,1,1), -1)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)

        try:
            bg[y:y+h, x:x+w] = bg[y:y+h, x:x+w] * (1 - mask) + cell * mask
        except:
            pass

        return bg

class CellTracker:
    def __init__(self, fps):
        self.trajectories = []
        self.fps = fps
        os.makedirs('data/output', exist_ok=True)

    def add_position(self, frame_idx, cell_id, x, y):
        self.trajectories.append({
            'frame': frame_idx,
            'time': frame_idx / self.fps,
            'cell_id': cell_id,
            'x': x,
            'y': y
        })

    def save_csv(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'time', 'cell_id', 'x', 'y'])
            writer.writeheader()
            writer.writerows(self.trajectories)

    def visualize_trajectories(self, background, output_file):
        img = background.copy()
        colors = (plt.cm.get_cmap('tab20', 20).colors * 255).astype(np.uint8)

        for cell_id in set(t['cell_id'] for t in self.trajectories):
            points = [(int(t['x']), int(t['y'])) for t in self.trajectories if t['cell_id'] == cell_id]
            if len(points) < 2:
                continue
            color = colors[cell_id % len(colors)]
            cv2.polylines(img, [np.array(points)], False, color.tolist(), 2)
            cv2.circle(img, points[0], 5, color.tolist(), -1)
            cv2.circle(img, points[-1], 5, color.tolist(), -1)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Cell Trajectories')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

def main():
    bg_folder = "data/fon"
    cell_folder = "data/patch"
    output_file = "data/output/cell_animation.mp4"
    duration = 10
    fps = 24
    num_cells = 12

    generator = CellAnimationGenerator()
    generator.load_assets(bg_folder, cell_folder)

    tracker = CellTracker(fps)
    canvas_size = (450, 450)
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, canvas_size)

    background = generator.generate_background(canvas_size)
    active_cells = []
    total_frames = duration * fps
    next_cell_id = 0

    for frame_idx in tqdm(range(total_frames), desc="Generating Animation"):
        frame = background.copy()
        t = frame_idx / fps

        if len(active_cells) < num_cells and frame_idx % (fps//2) == 0:
            cell_img = random.choice(generator.cell_patches)
            size = random.randint(40, 80)
            cell_img = cv2.resize(cell_img, (size, size))
            active_cells.append({
                'id': next_cell_id,
                'image': cell_img,
                'x0': random.uniform(0, canvas_size[0]),
                'y0': random.uniform(0, canvas_size[1]),
                'velocity': random.uniform(5, 15),
                'amplitude': random.uniform(10, 30),
                'frequency': random.uniform(0.05, 0.2),
                'noise_level': random.uniform(1, 3),
                'angle': random.uniform(0, 2 * math.pi),
                'life': 0
            })
            next_cell_id += 1

        for cell in active_cells[:]:
            cell['life'] += 1 / fps
            dx = cell['velocity'] * t
            dy = (cell['amplitude'] * math.sin(cell['frequency'] * dx) +
                  cell['noise_level'] * math.sin(t * 5 + cell['angle']))
            x = cell['x0'] + dx
            y = cell['y0'] + dy

            if (x < -100 or x > canvas_size[0]+100 or
                y < -100 or y > canvas_size[1]+100 or
                cell['life'] > duration * 0.8):
                active_cells.remove(cell)
                continue

            frame = generator.overlay_cell(frame, cell['image'], (x, y))
            tracker.add_position(frame_idx, cell['id'], x, y)

        video_writer.write(frame)

    video_writer.release()
    tracker.save_csv("data/output/trajectories.csv")
    tracker.visualize_trajectories(background, "data/output/trajectories.png")
