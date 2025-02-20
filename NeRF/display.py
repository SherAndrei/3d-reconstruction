#!/usr/bin/env python3

import keras
import numpy as np
import matplotlib.pyplot as plt

def display(images):
  num_images = images.shape[0]

  fig, axs = plt.subplots(2, 3)

  def display_random_pictures():
    for ax in axs.flat:
      random_index = np.random.randint(low=0, high=num_images)
      ax.imshow(images[random_index])
      ax.axis('off')  # Hide axes for better visualization

    plt.draw()

  def on_key_press(event):
    display_random_pictures()

  fig.canvas.mpl_connect('key_press_event', on_key_press)
  display_random_pictures()
  plt.tight_layout()  # Adjust layout for better spacing
  plt.show()

if __name__ == '__main__':
  url = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
  data = keras.utils.get_file(origin=url)

  data = np.load(data)
  display(data['images'])

