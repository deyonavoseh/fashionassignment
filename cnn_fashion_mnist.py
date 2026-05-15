"""
Fashion MNIST CNN Classifier — Python Implementation
A 6-layer Convolutional Neural Network implemented in Keras/TensorFlow
to classify grayscale clothing images across 10 categories.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF INFO/WARNING logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Fashion MNIST class labels (10 categories)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ── DATA PIPELINE ─────────────────────────────────────────────────
def load_data():
    """
    Loads and preprocesses the Fashion MNIST dataset.

    Preprocessing steps:
      - Reshape: (N, 28, 28) → (N, 28, 28, 1) to satisfy Conv2D input requirements
      - Normalise: scale pixel intensities from [0, 255] to [0, 1] for
        faster convergence and numerical stability during gradient descent
      - One-hot encode labels for compatibility with categorical cross-entropy loss
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test  = x_test .reshape(-1, 28, 28, 1).astype("float32") / 255.0

    y_train_ohe = to_categorical(y_train, num_classes=10)
    y_test_ohe  = to_categorical(y_test,  num_classes=10)

    print(f"Training samples : {x_train.shape[0]}")
    print(f"Test samples     : {x_test.shape[0]}")

    return (x_train, y_train, y_train_ohe), (x_test, y_test, y_test_ohe)


# ── MODEL ARCHITECTURE ────────────────────────────────────────────
class FashionCNN:
    """
    Six-layer CNN for multi-class image classification on Fashion MNIST.

    Architecture rationale:
      - Three successive Conv2D blocks progressively extract hierarchical
        spatial features: low-level edges → mid-level textures → high-level
        semantic patterns (e.g. collar shape, sole geometry)
      - MaxPooling after layers 1 and 2 provides spatial down-sampling,
        reducing parameter count and introducing translational invariance
      - A 256-unit Dense layer acts as a non-linear classifier over the
        flattened feature maps
      - Dropout (p=0.5) regularises the network and mitigates overfitting
      - Softmax activation on the output layer yields a valid probability
        distribution across the 10 classes
    """

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build()

    def _build(self) -> keras.Model:
        model = models.Sequential([

            # Layer 1 — Initial feature extraction
            # 32 filters capture low-frequency features (edges, gradients).
            # padding='same' preserves spatial dimensions post-convolution.
            # MaxPooling reduces spatial resolution: 28x28 → 14x14
            layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                          input_shape=self.input_shape, name="conv_1"),
            layers.MaxPooling2D((2, 2), name="pool_1"),

            # Layer 2 — Mid-level feature extraction
            # Doubling filters to 64 allows the network to learn richer,
            # more discriminative representations at a reduced spatial scale.
            # MaxPooling reduces spatial resolution: 14x14 → 7x7
            layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                          name="conv_2"),
            layers.MaxPooling2D((2, 2), name="pool_2"),

            # Layer 3 — High-level semantic feature extraction
            # 128 filters at 7x7 resolution encode abstract class-specific
            # patterns. No pooling here to retain spatial detail before flattening.
            layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                          name="conv_3"),

            # Layer 4 — Flatten
            # Converts the 3D feature tensor (7x7x128 = 6272-dim) into a
            # 1D vector for input to the fully connected classifier head.
            layers.Flatten(name="flatten"),

            # Layer 5 — Fully connected classifier
            # Dense(256) learns a non-linear combination of extracted features.
            # Dropout(0.5) randomly zeroes activations during training,
            # acting as an ensemble regulariser to reduce co-adaptation.
            layers.Dense(256, activation="relu", name="dense_1"),
            layers.Dropout(0.5, name="dropout"),

            # Layer 6 — Output layer
            # Softmax normalises logits into a probability distribution P(class|image).
            # The predicted class is argmax over these 10 probabilities.
            layers.Dense(self.num_classes, activation="softmax", name="output"),
        ])

        return model

    def compile(self):
        """
        Compiles the model with Adam optimiser and categorical cross-entropy loss.

        Adam is chosen for its adaptive learning rate per parameter, which
        generally outperforms vanilla SGD on vision tasks without requiring
        extensive LR tuning. Categorical cross-entropy is the standard loss
        for one-hot encoded multi-class classification.
        """
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model.summary()

    def train(self, x_train, y_train, epochs=10, batch_size=64, validation_split=0.1):
        """
        Trains the model. 10% of training data is held out as a validation
        set to monitor generalisation and detect overfitting per epoch.
        """
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history

    def evaluate(self, x_test, y_test):
        """Evaluates classification performance on the held-out test set."""
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest Accuracy : {acc:.4f}")
        print(f"Test Loss     : {loss:.4f}")
        return loss, acc

    def predict(self, images):
        """Returns softmax probability vectors for each input image."""
        return self.model.predict(images)

    def save(self, path="fashion_cnn.keras"):
        """Serialises the trained model weights and architecture to disk."""
        self.model.save(path)
        print(f"Model saved → {path}")


# ── TRAINING DIAGNOSTICS ──────────────────────────────────────────
def plot_history(history):
    """
    Plots training vs. validation accuracy and loss curves.
    A growing gap between train and val curves indicates overfitting.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Training curves saved → training_history.png")


# ── INFERENCE ON SAMPLE IMAGES ────────────────────────────────────
def predict_samples(cnn, x_test, y_test, indices=(0, 1, 2, 3)):
    """
    Runs inference on selected test images and visualises results.
    Predicted class = argmax of the softmax output vector.
    Title colour: green = correct prediction, red = misclassification.
    """
    images = x_test[list(indices)]
    labels = y_test[list(indices)]
    preds  = cnn.predict(images)

    fig, axes = plt.subplots(1, len(indices), figsize=(14, 4))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(28, 28), cmap="gray")
        true_label = CLASS_NAMES[labels[i]]
        pred_label = CLASS_NAMES[np.argmax(preds[i])]
        confidence = np.max(preds[i]) * 100
        color = "green" if true_label == pred_label else "red"
        ax.set_title(
            f"True:  {true_label}\nPred:  {pred_label}\nConf: {confidence:.1f}%",
            color=color, fontsize=9
        )
        ax.axis("off")

    plt.suptitle("Fashion MNIST — Inference Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150)
    plt.show()
    print("Predictions saved → sample_predictions.png")


# ── ENTRY POINT ───────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load and preprocess data
    (x_train, y_train, y_train_ohe), (x_test, y_test, y_test_ohe) = load_data()

    # 2. Instantiate and compile the CNN
    cnn = FashionCNN()
    cnn.compile()

    # 3. Train for 10 epochs
    history = cnn.train(x_train, y_train_ohe, epochs=10, batch_size=64)

    # 4. Evaluate on the test set
    cnn.evaluate(x_test, y_test_ohe)

    # 5. Plot training diagnostics
    plot_history(history)

    # 6. Run inference on 4 sample test images
    predict_samples(cnn, x_test, y_test, indices=(0, 1, 2, 3))

    # 7. Persist the trained model
    cnn.save()