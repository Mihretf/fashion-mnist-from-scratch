import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

TRAIN_PATH = "datasets/fashion-mnist_train.csv"

MASK_SIZE = 8
NUM_SAMPLES = 10

RANDOM_SEED = 42


# =========================
# Load data
# =========================

train = pd.read_csv(TRAIN_PATH)

pixel_columns = train.columns[1:]

images = train[pixel_columns].values.astype(np.float32)


# =========================
# Create masked images
# =========================

rng = np.random.default_rng(RANDOM_SEED)

masked_images = []
mask_positions = []

for i in range(NUM_SAMPLES):

    image = images[i].reshape(28, 28).copy()

    # Random top-left corner for the mask
    row = rng.integers(0, 28 - MASK_SIZE + 1)
    col = rng.integers(0, 28 - MASK_SIZE + 1)

    # Save the original pixels that will be hidden
    original_region = image[
        row:row + MASK_SIZE,
        col:col + MASK_SIZE
    ].copy()

    # Hide the region
    image[
        row:row + MASK_SIZE,
        col:col + MASK_SIZE
    ] = 0

    masked_images.append(image)
    mask_positions.append((row, col, original_region))


# =========================
# Visualize masking
# =========================

fig, axes = plt.subplots(
    2,
    NUM_SAMPLES,
    figsize=(15, 4)
)

for i in range(NUM_SAMPLES):

    # Original image
    original = images[i].reshape(28, 28)

    axes[0, i].imshow(original, cmap="gray")
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    # Masked image
    axes[1, i].imshow(masked_images[i], cmap="gray")
    axes[1, i].set_title("Masked")
    axes[1, i].axis("off")


axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("Masked", fontsize=10)

plt.tight_layout()

plt.savefig(
    "experiments/masking_examples.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# =========================
# Verify masking
# =========================

print("========== MASKING ==========")
print(f"Mask size: {MASK_SIZE} x {MASK_SIZE}")
print(f"Number of examples: {NUM_SAMPLES}")

print("\nMask positions:")

for i, (row, col, original_region) in enumerate(mask_positions):
    print(
        f"Image {i}: "
        f"row={row}, col={col}, "
        f"hidden pixels={original_region.size}"
    )

print("\nMasking complete.")