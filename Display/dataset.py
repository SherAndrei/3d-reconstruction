#!/usr/bin/env python3
"""
Display images in a grid layout with pagination.

The dataset NPZ file must contain an 'images' array with shape
(num_images, width, height, 3).

Usage:
    python dataset.py /path/to/dataset.npz [--grid_size 5]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Display images in a grid layout with pagination."
    )
    parser.add_argument(
        "dataset",
        help="Path to the NPZ dataset file (must contain an 'images' array).",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=5,
        help="Number of images per row/column in the grid (default: 5).",
    )
    return parser.parse_args()


def display_square(images, grid_size):
    """
    Display images in a grid layout with pagination.

    Parameters:
        images (np.ndarray): Array of images to display, shape (N, W, H, 3).
        grid_size (int): The number of images per row/column in the grid.
    """
    if len(images.shape) != 4:
        raise ValueError("Expected images with shape (num_images, width, height, 3)")

    num_images = images.shape[0]
    images_on_page = grid_size * grid_size
    num_pages = (num_images + images_on_page - 1) // images_on_page
    current_page = 0

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))

    def update_display(page_idx):
        nonlocal current_page
        current_page = page_idx
        start = page_idx * images_on_page
        end = start + images_on_page
        current_images = images[start:end]
        for ax, img in zip(axs.flat, current_images):
            ax.clear()
            ax.axis("off")
            ax.imshow(img)
        # Clear remaining axes if any.
        for ax in axs.flat[len(current_images):]:
            ax.clear()
            ax.axis("off")
        plt.draw()

    def on_key_press(event):
        nonlocal current_page
        if event.key == "right":
            new_page = (current_page + 1) % num_pages
            update_display(new_page)
        elif event.key == "left":
            new_page = (current_page - 1) % num_pages
            update_display(new_page)
        fig.canvas.draw_idle()

    fig.canvas.manager.set_window_title("Dataset Showcase")
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    update_display(0)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    data = np.load(args.dataset)
    if "images" not in data:
        raise KeyError("The dataset file must contain an 'images' array.")
    images = data["images"]
    display_square(images, args.grid_size)


if __name__ == "__main__":
    main()
