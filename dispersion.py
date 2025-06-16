import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm 
from pathlib import Path

def dispersion(points, save_path = None):
    N, d = points.shape
    assert d == 2, "Plotting only for 2D"

    # Get coordinate grid
    grid_coords = [np.unique(np.concatenate(([0, 1], points[:, j]))) for j in range(d)]
    grid_lengths = [len(g) for g in grid_coords]

    max_volume = 0.0
    best_box = None

    # Precompute grid for speed
    grid_coords = [np.array(g) for g in grid_coords]

    for lower_idx in tqdm(product(*[range(L - 1) for L in grid_lengths]), desc = "Computing Dispersion"):
        for upper_idx in product(*[range(i + 1, L) for i, L in zip(lower_idx, grid_lengths)]):


            box_min = np.array([grid_coords[j][lower_idx[j]] for j in range(d)])
            box_max = np.array([grid_coords[j][upper_idx[j]] for j in range(d)])

            # reject zero-vol boxes
            if np.any(box_min >= box_max):
                continue

            # check: point in box? (Vectorized) point-in-box check
            mask = np.all((points > box_min) & (points < box_max), axis=1)

            if not np.any(mask):
                volume = np.prod(box_max - box_min)
                if volume > max_volume:
                    max_volume = volume
                    best_box = (box_min.copy(), box_max.copy())

    print(f"Dispersion: {max_volume}")
    print(f"Biggest Empty Box lower corner: {best_box[0]}")
    print(f"Biggest Empty Box upper corner: {best_box[1]}")

    # Plot the points and the best box
    if d == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(points[:, 0], points[:, 1], c='blue', s=10, label='Points')
        rect = plt.Rectangle(best_box[0], *(best_box[1] - best_box[0]),
                             edgecolor='red', facecolor='none', lw=2, label='Biggest Empty box')
        plt.gca().add_patch(rect)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.title("Dispersion and Largest Empty Box")
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents = True, exist_ok = True)
            plt.savefig(save_path, bbox_inches = 'tight')
        plt.close()

    return max_volume, best_box


"""
Focus: 
1. how did they loop through all boxes (faster than my method)
2. np vs tensor input
"""