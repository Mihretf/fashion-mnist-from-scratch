import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

TRAIN_PATH = "datasets/fashion-mnist_train.csv"
TEST_PATH = "datasets/fashion-mnist_test.csv"

PIXEL_COLUMNS = [str(i) for i in range(784)]

LABEL_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


# =========================
# Load data
# =========================

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


# =========================
# Basic inspection
# =========================

print("========== DATASET SHAPES ==========")
print(f"Training set: {train.shape}")
print(f"Test set:     {test.shape}")


print("\n========== COLUMNS ==========")
print(f"Number of columns: {len(train.columns)}")
print(f"First 10 columns: {list(train.columns[:10])}")
print(f"Last 5 columns:   {list(train.columns[-5:])}")


# =========================
# Missing values
# =========================

print("\n========== MISSING VALUES ==========")
print(f"Training set missing values: {train.isnull().sum().sum()}")
print(f"Test set missing values:     {test.isnull().sum().sum()}")


# =========================
# Duplicate rows
# =========================

print("\n========== DUPLICATES ==========")
print(f"Training set duplicate rows: {train.duplicated().sum()}")
print(f"Test set duplicate rows:     {test.duplicated().sum()}")


# =========================
# Separate labels and pixels
# =========================

label_column = train.columns[0]
pixel_columns = train.columns[1:]

X_train = train[pixel_columns]
y_train = train[label_column]

X_test = test[pixel_columns]
y_test = test[label_column]


# =========================
# Pixel information
# =========================

print("\n========== PIXEL INFORMATION ==========")
print(f"Number of pixel columns: {len(pixel_columns)}")
print(f"Expected pixels for 28x28 image: {28 * 28}")

print(f"\nMinimum pixel value: {X_train.min().min()}")
print(f"Maximum pixel value: {X_train.max().max()}")

print(f"Training pixel dtype: {X_train.dtypes.iloc[0]}")


# =========================
# Class distribution
# =========================

print("\n========== CLASS DISTRIBUTION ==========")

class_counts = y_train.value_counts().sort_index()

for label, count in class_counts.items():
    class_name = LABEL_NAMES.get(label, "Unknown")
    percentage = (count / len(y_train)) * 100

    print(
        f"{label}: {class_name:<12} "
        f"{count:>6} images "
        f"({percentage:.2f}%)"
    )


# =========================
# Visualize sample images
# =========================

print("\n========== VISUALIZING SAMPLE IMAGES ==========")

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    image = X_train.iloc[i].values.astype(np.uint8)
    image = image.reshape(28, 28)

    label = y_train.iloc[i]
    class_name = LABEL_NAMES.get(label, "Unknown")

    ax.imshow(image, cmap="gray")
    ax.set_title(f"{label}: {class_name}")
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "experiments/sample_images.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# =========================
# Final summary
# =========================

print("\n========== SUMMARY ==========")
print(f"Training examples: {len(train)}")
print(f"Test examples:     {len(test)}")
print(f"Image size:        28 x 28")
print(f"Pixels per image:  {len(pixel_columns)}")
print(f"Number of classes: {y_train.nunique()}")
print(f"Pixel range:       {X_train.min().min()} - {X_train.max().max()}")

print("\nDataset inspection complete.")