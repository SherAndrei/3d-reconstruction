#!/usr/bin/env python3
"""
Display NeRF dataset poses, image, and camera view.

Usage:
    python display_dataset.py /path/to/dataset.npz

The script loads the NPZ dataset (with keys: images, poses, focal) and displays:
  - The 4x4 camera pose matrix,
  - The corresponding image,
  - A 3D view showing the camera, object, and view rays.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_rays(height, width, focal, pose):
    """Computes origin point and direction vector of rays.

    Args:
        height: Height of the image.
        width: Width of the image.
        focal: The focal length between the images and the camera.
        pose: The pose matrix of the camera.

    Returns:
        Tuple of origin point and direction vector for rays.
    """
    # Build a meshgrid for the rays.
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )

    # Normalize the x axis coordinates.
    transformed_i = (i - width * 0.5) / focal

    # Normalize the y axis coordinates.
    transformed_j = (j - height * 0.5) / focal

    # Create the direction unit vectors.
    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)

    # Get the camera matrix.
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    # Get origins and directions for the rays.
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))

    # Return the origins and directions.
    return (ray_origins, ray_directions)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Display NeRF dataset poses, image, and camera view."
    )
    parser.add_argument(
        "dataset",
        help="Path to the NPZ dataset file (e.g., tiny_nerf_data.npz).",
    )
    return parser.parse_args()


def display_matrix(ax, pose_matrix):
    """Display the 4x4 camera pose matrix on the given axis."""
    ax.clear()
    matrix_str = "\n".join(
        " ".join(f"{val:7.3f}" for val in row) for row in pose_matrix
    )
    ax.text(
        0.5,
        0.5,
        matrix_str,
        fontsize=10,
        fontfamily="monospace",
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.axis("off")
    ax.set_title("Camera Pose Matrix (4x4)")


def display_image(ax, image):
    """Display the image on the given axis."""
    ax.clear()
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Camera Image")


def display_scene(ax, poses, idx, num_images, height, width, focal):
    """Display the 3D scene with camera, object, and view rays."""
    ax.clear()
    ax.set_title("3D Scene")

    all_cam_positions = poses[:, :3, 3]
    x_lim = (
        -np.max(np.abs(all_cam_positions[:, 0])) - 1,
        np.max(np.abs(all_cam_positions[:, 0])) + 1,
    )
    y_lim = (
        -np.max(np.abs(all_cam_positions[:, 1])) - 1,
        np.max(np.abs(all_cam_positions[:, 1])) + 1,
    )
    z_lim = (0, np.max(np.abs(all_cam_positions[:, 2])) + 1)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    ax.scatter(0, 0, 0, color="red", s=100, label="Object")

    cam_pos = poses[idx][:3, 3]
    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2],
               color="yellow", s=50, label="Camera")

    pose_tf = tf.convert_to_tensor(poses[idx], dtype=tf.float32)
    ray_origins, ray_directions = get_rays(height, width, focal, pose_tf)

    def display_ray(y, x):
        origin_center = ray_origins[y, x].numpy()
        direction_center = ray_directions[y, x].numpy()
        scale = 2.0
        end_point = origin_center + direction_center * scale
        ax.plot(
            [origin_center[0], end_point[0]],
            [origin_center[1], end_point[1]],
            [origin_center[2], end_point[2]],
            color="green",
            linewidth=2,
        )

    # Plot rays along image borders.
    for y in range(0, height, 2):
        display_ray(y, 0)
        display_ray(y, width - 1)
    for x in range(0, width, 2):
        display_ray(0, x)
        display_ray(height - 1, x)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    info_text = (
        f"Frame: {idx+1}/{num_images}\n"
        f"Focal Length: {focal:.2f}\n"
        f"Camera Position: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})\n"
        "Object Position: (0.00, 0.00, 0.00)"
    )
    ax.text2D(
        0.05, 0.95, info_text, transform=ax.transAxes,
        fontsize=9, verticalalignment="top"
    )
    ax.legend()


def update_display(idx, num_images, height, width, focal, images, poses,
                   ax_matrix, ax_image, ax_scene):
    """Update the display for the given frame index."""
    display_matrix(ax_matrix, poses[idx])
    display_image(ax_image, images[idx])
    display_scene(ax_scene, poses, idx, num_images, height, width, focal)


def on_key(event, num_images, height, width, focal, images, poses,
           ax_matrix, ax_image, ax_scene, fig):
    """Key event handler for updating the display on left/right arrow keys."""
    global current_index
    if event.key == "right":
        new_index = (current_index + 1) % num_images
        current_index = new_index
        update_display(new_index, num_images, height, width, focal,
                       images, poses, ax_matrix, ax_image, ax_scene)
        fig.canvas.draw_idle()
    elif event.key == "left":
        new_index = (current_index - 1) % num_images
        current_index = new_index
        update_display(new_index, num_images, height, width, focal,
                       images, poses, ax_matrix, ax_image, ax_scene)
        fig.canvas.draw_idle()


def main():
    args = parse_args()
    dataset_path = args.dataset

    # Load dataset from NPZ file.
    data = np.load(dataset_path)
    images = data["images"]
    poses = data["poses"]
    focal = float(data["focal"])

    num_images, H, W, _ = images.shape

    # Create figure and subplots.
    fig = plt.figure(figsize=(12, 4))
    ax_matrix = fig.add_subplot(1, 3, 1)
    ax_image = fig.add_subplot(1, 3, 2)
    ax_scene = fig.add_subplot(1, 3, 3, projection="3d")

    global current_index
    current_index = 0
    update_display(current_index, num_images, H, W, focal, images, poses,
                   ax_matrix, ax_image, ax_scene)

    fig.canvas.manager.set_window_title("Poses Showcase")
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key(event, num_images, H, W, focal, images, poses,
                             ax_matrix, ax_image, ax_scene, fig)
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
