import os
import numpy as np
import matplotlib.pyplot as plt

from model import MaskedImageModel


# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = "datasets/fashion-mnist_train.csv"
TEST_PATH = "datasets/fashion-mnist_test.csv"

MASK_SIZES = [8, 10, 12, 14, 16]

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

NUM_VISUAL_EXAMPLES = 10

SEED = 42


# ============================================================
# Load data
# ============================================================

train_data = np.loadtxt(
    TRAIN_PATH,
    delimiter=",",
    skiprows=1
)

test_data = np.loadtxt(
    TEST_PATH,
    delimiter=",",
    skiprows=1
)

train_images = train_data[:TRAINING_EXAMPLES, 1:] / 255.0
test_images = test_data[:TESTING_EXAMPLES, 1:] / 255.0

print("Training images:", train_images.shape)
print("Testing images:", test_images.shape)


# ============================================================
# Run experiment for each mask size
# ============================================================

for mask_size in MASK_SIZES:

    print("\n" + "=" * 60)
    print(f"MASK SIZE: {mask_size} x {mask_size}")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    output_size = mask_size * mask_size

    # --------------------------------------------------------
    # Create training masks
    # --------------------------------------------------------

    X_train = []
    y_train = []

    for image in train_images:

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

        X_train.append(masked.flatten())
        y_train.append(target.flatten())

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(
        "Training data:",
        X_train.shape,
        "Targets:",
        y_train.shape
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MaskedImageModel(
        input_size=784,
        hidden_size=128,
        output_size=output_size,
        seed=SEED
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    num_samples = X_train.shape[0]

    for epoch in range(EPOCHS):

        indices = rng.permutation(num_samples)

        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0
        num_batches = 0

        for start in range(
            0,
            num_samples,
            BATCH_SIZE
        ):

            end = start + BATCH_SIZE

            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            predictions = model.forward(X_batch)

            loss = model.mse_loss(
                predictions,
                y_batch
            )

            model.backward(
                X_batch,
                y_batch,
                learning_rate=LEARNING_RATE
            )

            epoch_loss += loss
            num_batches += 1

        epoch_loss /= num_batches

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {epoch_loss:.6f}"
        )

    # --------------------------------------------------------
    # Create test masks
    # --------------------------------------------------------

    test_rng = np.random.default_rng(SEED)

    X_test = []
    y_test = []
    positions = []

    selected_test_images = test_images[
        :NUM_VISUAL_EXAMPLES
    ]

    for image in selected_test_images:

        image_2d = image.reshape(28, 28)

        row = test_rng.integers(
            0,
            28 - mask_size + 1
        )

        col = test_rng.integers(
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

        X_test.append(masked.flatten())
        y_test.append(target.flatten())
        positions.append((row, col))

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    predictions = model.forward(X_test)

    predictions = np.clip(
        predictions,
        0,
        1
    )

    test_mse = np.mean(
        (predictions - y_test) ** 2
    )

    print(
        f"\nTest reconstruction MSE: "
        f"{test_mse:.6f}"
    )

    # --------------------------------------------------------
    # Save experiment
    # --------------------------------------------------------

    output_dir = (
        f"experiments/{mask_size}x{mask_size}_mask"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Save model weights
    np.savez(
        f"{output_dir}/model_weights.npz",
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2
    )

    # Save MSE
    with open(
        f"{output_dir}/results.txt",
        "w"
    ) as file:

        file.write(
            f"Mask size: {mask_size}x{mask_size}\n"
        )

        file.write(
            f"Test reconstruction MSE: "
            f"{test_mse:.6f}\n"
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        NUM_VISUAL_EXAMPLES,
        3,
        figsize=(6, 20)
    )

    for i in range(NUM_VISUAL_EXAMPLES):

        row, col = positions[i]

        original = selected_test_images[i].reshape(
            28,
            28
        )

        masked = X_test[i].reshape(
            28,
            28
        )

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

        axes[i, 0].set_title(
            "Original"
        )

        axes[i, 1].imshow(
            masked,
            cmap="gray"
        )

        axes[i, 1].set_title(
            "Masked"
        )

        axes[i, 2].imshow(
            reconstructed,
            cmap="gray"
        )

        axes[i, 2].set_title(
            "Reconstructed"
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/reconstruction.png"
    )

    plt.close()

    print(
        f"Saved results to: {output_dir}"
    )


print("\n" + "=" * 60)
print("MASK SIZE EXPERIMENT COMPLETE")
print("=" * 60)