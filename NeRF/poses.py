import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

URL = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
data_path = keras.utils.get_file(origin=URL)
data = np.load(data_path)
images = data["images"]
poses = data["poses"]
focal = float(data["focal"])

num_images, H, W, _ = images.shape


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
    # i.shape (height, width)
    # j.shape (width, height)

    # Normalize the x axis coordinates.
    transformed_i = (i - width * 0.5) / focal

    # Normalize the y axis coordinates.
    transformed_j = (j - height * 0.5) / focal

    # Create the direction unit vectors.
    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)
    # directions.shape (height, width, 3)

    # Get the camera matrix.
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    # Get origins and directions for the rays.
    transformed_dirs = directions[..., None, :]
    # transformed_dirs.shape (height, width, 1, 3)
    camera_dirs = transformed_dirs * camera_matrix
    # camera_dirs.shape (height, width, 3, 3)
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1)
    # ray_directions.shape (height, width, 3)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))
    # ray_origins.shape (height, width, 3)

    # Return the origins and directions.
    return (ray_origins, ray_directions)


fig = plt.figure(figsize=(12, 4))
ax_matrix = fig.add_subplot(1, 3, 1)
ax_image = fig.add_subplot(1, 3, 2)
ax_scene = fig.add_subplot(1, 3, 3, projection='3d')


def display_matrix(ax, pose_matrix):
    ax.clear()
    matrix_str = "\n".join(
        " ".join(f"{val:7.3f}" for val in row) for row in pose_matrix)
    ax.text(0.5, 0.5, matrix_str,
            fontsize=10,
            fontfamily='monospace',
            horizontalalignment='center',
            verticalalignment='center')
    ax.axis('off')
    ax.set_title("Camera Pose Matrix (4x4)")


def display_image(ax, image):
    ax.clear()
    ax.imshow(image)
    ax.axis('off')
    ax.set_title("Camera Image")


all_cam_positions = poses[:, :3, 3]
x_lim = (-np.max(np.abs(all_cam_positions[:, 0])) - 1,
         np.max(np.abs(all_cam_positions[:, 0])) + 1)
y_lim = (-np.max(np.abs(all_cam_positions[:, 1])) - 1,
         np.max(np.abs(all_cam_positions[:, 1])) + 1)
z_lim = (0, np.max(np.abs(all_cam_positions[:, 2])) + 1)


def display_scene(ax, poses, idx):
    ax.clear()
    ax.set_title("3D Scene")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    ax.scatter(0, 0, 0, color='red', s=100, label='Object')

    cam_pos = poses[idx][:3, 3]
    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2],
               color='yellow', s=50, label='Camera')

    pose_tf = tf.convert_to_tensor(poses[idx], dtype=tf.float32)
    ray_origins, ray_directions = get_rays(H, W, focal, pose_tf)

    def display_ray(y, x):
        origin_center = ray_origins[y, x].numpy()
        direction_center = ray_directions[y, x].numpy()

        scale = 2.0
        end_point = origin_center + direction_center * scale
        ax.plot([origin_center[0], end_point[0]],
                [origin_center[1], end_point[1]],
                [origin_center[2], end_point[2]],
                color='green', linewidth=2)

    # plot only frame of the image
    for y in range(0, H, 2):
        display_ray(y, 0)
        display_ray(y, W - 1)

    for x in range(0, W, 2):
        display_ray(0, x)
        display_ray(H - 1, x)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    info_text = (
        f"Frame: {idx+1}/{num_images}\n"
        f"Focal Length: {focal:.2f}\n"
        f"Camera Position: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})\n"
        f"Object Position: (0.00, 0.00, 0.00)"
    )
    ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes,
              fontsize=9, verticalalignment='top')

    ax.legend()


current_index = 0


def update_display(idx):
    global current_index
    current_index = idx

    display_matrix(ax_matrix, poses[idx])
    display_image(ax_image, images[idx])
    display_scene(ax_scene, poses, idx)


update_display(current_index)


def on_key(event):
    global current_index
    if event.key == 'right':
        new_index = (current_index + 1) % num_images
        update_display(new_index)
        fig.canvas.draw_idle()
    elif event.key == 'left':
        new_index = (current_index - 1) % num_images
        update_display(new_index)
        fig.canvas.draw_idle()


fig.canvas.manager.set_window_title('Poses showcase')
fig.canvas.mpl_connect('key_press_event', on_key)
plt.tight_layout()
plt.show()
