import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import MaskedImageModel


# =========================
# Configuration
# =========================

TRAIN_PATH = "datasets/fashion-mnist_train.csv"

MASK_SIZE = 8
NUM_TRAIN_SAMPLES = 10000

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

RANDOM_SEED = 42


# =========================
# Load dataset
# =========================

print("Loading dataset...")

train = pd.read_csv(TRAIN_PATH)

pixels = train.iloc[:, 1:].values.astype(np.float32)

# Normalize pixel values from 0–255 to 0–1
pixels /= 255.0

print(f"Loaded {len(pixels)} images.")


# =========================
# Create masked dataset
# =========================

print("Creating masked training data...")

rng = np.random.default_rng(RANDOM_SEED)

images = pixels[:NUM_TRAIN_SAMPLES]

X = []
y = []

for image in images:

    image_2d = image.reshape(28, 28).copy()

    # Random mask location
    row = rng.integers(0, 28 - MASK_SIZE + 1)
    col = rng.integers(0, 28 - MASK_SIZE + 1)

    # Save the original hidden pixels
    target = image_2d[
        row:row + MASK_SIZE,
        col:col + MASK_SIZE
    ].copy()

    # Hide the region
    image_2d[
        row:row + MASK_SIZE,
        col:col + MASK_SIZE
    ] = 0

    X.append(image_2d.flatten())
    y.append(target.flatten())


X = np.array(X)
y = np.array(y)

print(f"Input shape:  {X.shape}")
print(f"Target shape: {y.shape}")


# =========================
# Create model
# =========================

model = MaskedImageModel(
    input_size=784,
    hidden_size=128,
    output_size=64,
    seed=RANDOM_SEED
)


# =========================
# Training
# =========================

print("\n========== TRAINING ==========")

loss_history = []

num_samples = X.shape[0]

for epoch in range(EPOCHS):

    # Shuffle training examples
    indices = rng.permutation(num_samples)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    epoch_loss = 0.0
    num_batches = 0

    for start in range(0, num_samples, BATCH_SIZE):

        end = start + BATCH_SIZE

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Forward pass
        predictions = model.forward(X_batch)

        # Calculate loss
        loss = model.mse_loss(predictions, y_batch)

        # Backpropagation + parameter update
        model.backward(
            X_batch,
            y_batch,
            learning_rate=LEARNING_RATE
        )

        epoch_loss += loss
        num_batches += 1

    average_loss = epoch_loss / num_batches

    loss_history.append(average_loss)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Loss: {average_loss:.6f}"
    )


# =========================
# Plot training loss
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    loss_history,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Masked Image Reconstruction Training Loss")

plt.grid(True)

plt.savefig(
    "experiments/training_loss.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


print("\nTraining complete.")

np.savez(
    "experiments/model_weights.npz",
    W1=model.W1,
    b1=model.b1,
    W2=model.W2,
    b2=model.b2
)

print("Model weights saved.")