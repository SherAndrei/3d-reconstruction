#!/usr/bin/env python3

import keras
import numpy as np
import matplotlib.pyplot as plt

def display_square(images, grid_size):
    """
    Display images in a grid layout with pagination.
    
    Parameters:
        images (np.ndarray): Array of images to display.
        grid_size (int): The number of images per row/column in the grid.
    """

    assert len(images.shape) == 4 # expect (num_images, width, height, rgb)

    current_page = 0
    num_images = images.shape[0]
    images_on_page = grid_size * grid_size
    num_pages = (num_images + images_on_page - 1) // images_on_page

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))

    def update_display(idx):
        nonlocal current_page
        current_page = idx
        start = idx * images_on_page
        end = start + images_on_page
        current_images = images[start:end]
        for ax, img in zip(axs.flat, current_images):
            ax.clear()
            ax.axis('off')
            ax.imshow(img)
        for ax in axs.flat[len(current_images):]:
            ax.clear()
            ax.axis('off')
        plt.draw()

    def on_key_press(event):
        nonlocal current_page
        if event.key == 'right':
            new_index = (current_page + 1) % num_pages
            update_display(new_index)
        elif event.key == 'left':
            new_index = (current_page - 1) % num_pages
            update_display(new_index)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    update_display(0)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    URL = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
    file_path = keras.utils.get_file(origin=URL)
    data = np.load(file_path)
    display_square(data['images'], 5)
