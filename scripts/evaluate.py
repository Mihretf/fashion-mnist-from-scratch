import os
import numpy as np
import matplotlib.pyplot as plt

from model import MaskedImageModel


# -----------------------------
# 1. Load test data
# -----------------------------

TEST_PATH = "datasets/fashion-mnist_test.csv"
WEIGHTS_PATH = "experiments/model_weights.npz"

data = np.loadtxt(TEST_PATH, delimiter=",", skiprows=1)

pixels = data[:, 1:]
images = pixels / 255.0

print("Test images:", images.shape)


# -----------------------------
# 2. Load trained model
# -----------------------------

model = MaskedImageModel()

weights = np.load(WEIGHTS_PATH)

model.W1 = weights["W1"]
model.b1 = weights["b1"]
model.W2 = weights["W2"]
model.b2 = weights["b2"]

print("Model weights loaded.")


# -----------------------------
# 3. Mask-size experiment
# -----------------------------

mask_sizes = [8, 10, 12, 14, 16]

num_examples = 10

rng = np.random.default_rng(123)


for mask_size in mask_sizes:

    print(f"\n{'=' * 50}")
    print(f"Mask size: {mask_size} x {mask_size}")
    print(f"{'=' * 50}")

    selected_images = images[:num_examples]

    masked_images = []
    targets = []
    positions = []

    # -----------------------------
    # Create masked images
    # -----------------------------

    for image in selected_images:

        image_2d = image.reshape(28, 28)

        row = rng.integers(
            0,
            28 - mask_size + 1
        )

        col = rng.integers(
            0,
            28 - mask_size + 1
        )

        masked = image_2d.copy()

        target = image_2d[
            row:row + mask_size,
            col:col + mask_size
        ].copy()

        masked[
            row:row + mask_size,
            col:col + mask_size
        ] = 0

        masked_images.append(masked.flatten())
        targets.append(target.flatten())
        positions.append((row, col))


    masked_images = np.array(masked_images)
    targets = np.array(targets)


    # -----------------------------
    # Reconstruct hidden pixels
    # -----------------------------

    predictions = model.forward(masked_images)

    predictions = np.clip(predictions, 0, 1)


    # -----------------------------
    # Calculate MSE
    # -----------------------------

    test_mse = np.mean(
        (predictions - targets) ** 2
    )

    print(
        f"Total reconstruction MSE: {test_mse:.6f}"
    )


    # -----------------------------
    # Save experiment
    # -----------------------------

    output_dir = (
        f"experiments/{mask_size}x{mask_size}_mask"
    )

    os.makedirs(output_dir, exist_ok=True)


    # -----------------------------
    # Visualize
    # -----------------------------

    fig, axes = plt.subplots(
        num_examples,
        3,
        figsize=(6, 20)
    )

    for i in range(num_examples):

        row, col = positions[i]

        original = selected_images[i].reshape(28, 28)

        masked = masked_images[i].reshape(28, 28)

        reconstructed = masked.copy()

        reconstructed[
            row:row + mask_size,
            col:col + mask_size
        ] = predictions[i].reshape(
            mask_size,
            mask_size
        )

        axes[i, 0].imshow(
            original,
            cmap="gray"
        )
        axes[i, 0].set_title("Original")

        axes[i, 1].imshow(
            masked,
            cmap="gray"
        )
        axes[i, 1].set_title("Masked")

        axes[i, 2].imshow(
            reconstructed,
            cmap="gray"
        )
        axes[i, 2].set_title("Reconstructed")

        for ax in axes[i]:
            ax.axis("off")


    plt.tight_layout()

    output_path = (
        f"{output_dir}/reconstruction.png"
    )

    plt.savefig(output_path)

    plt.close()

    print(f"Saved: {output_path}")