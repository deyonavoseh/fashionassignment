# ══════════════════════════════════════════════════════════════════
# Fashion MNIST CNN Classifier — R Implementation
# A 6-layer Convolutional Neural Network using Keras for R
# to classify grayscale clothing images across 10 categories.
# ══════════════════════════════════════════════════════════════════

library(keras3)


# ── CLASS LABELS ──────────────────────────────────────────────────
# Fashion MNIST contains 10 mutually exclusive clothing categories
class_names <- c(
  "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
)


# ── DATA PIPELINE ─────────────────────────────────────────────────
# Keras downloads and caches Fashion MNIST automatically on first run.
# Preprocessing steps:
#   - Reshape: (N, 28, 28) → (N, 28, 28, 1) to satisfy Conv2D input requirements
#   - Normalise: scale pixel intensities from [0, 255] to [0, 1] for
#     faster convergence and numerical stability during gradient descent
#   - One-hot encode labels for compatibility with categorical cross-entropy loss
fashion  <- dataset_fashion_mnist()

x_train  <- fashion$train$x
y_train  <- fashion$train$y
x_test   <- fashion$test$x
y_test   <- fashion$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1)) / 255
x_test  <- array_reshape(x_test,  c(nrow(x_test),  28, 28, 1)) / 255

# One-hot encode class labels
y_train_ohe <- to_categorical(y_train, 10)
y_test_ohe  <- to_categorical(y_test,  10)

cat("Training samples :", nrow(x_train), "\n")
cat("Test samples     :", nrow(x_test),  "\n")


# ── MODEL ARCHITECTURE ────────────────────────────────────────────
# Six-layer CNN for multi-class image classification on Fashion MNIST.
#
# Architecture rationale:
#   - Three successive Conv2D blocks progressively extract hierarchical
#     spatial features: low-level edges → mid-level textures → high-level
#     semantic patterns (e.g. collar shape, sole geometry)
#   - MaxPooling after layers 1 and 2 provides spatial down-sampling,
#     reducing parameter count and introducing translational invariance
#   - A 256-unit Dense layer acts as a non-linear classifier over the
#     flattened feature maps
#   - Dropout (p=0.5) regularises the network and mitigates overfitting
#   - Softmax activation on the output layer yields a valid probability
#     distribution across the 10 classes
model <- keras_model_sequential(name = "FashionCNN") %>%

  # Layer 1 — Initial feature extraction
  # 32 filters capture low-frequency features (edges, gradients).
  # padding='same' preserves spatial dimensions post-convolution.
  # MaxPooling reduces spatial resolution: 28x28 → 14x14
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                padding = "same", input_shape = c(28, 28, 1), name = "conv_1") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), name = "pool_1") %>%

  # Layer 2 — Mid-level feature extraction
  # Doubling filters to 64 allows the network to learn richer,
  # more discriminative representations at a reduced spatial scale.
  # MaxPooling reduces spatial resolution: 14x14 → 7x7
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu",
                padding = "same", name = "conv_2") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), name = "pool_2") %>%

  # Layer 3 — High-level semantic feature extraction
  # 128 filters at 7x7 resolution encode abstract class-specific
  # patterns. No pooling here to retain spatial detail before flattening.
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu",
                padding = "same", name = "conv_3") %>%

  # Layer 4 — Flatten
  # Converts the 3D feature tensor (7x7x128 = 6272-dim) into a
  # 1D vector for input to the fully connected classifier head.
  layer_flatten(name = "flatten") %>%

  # Layer 5 — Fully connected classifier
  # Dense(256) learns a non-linear combination of extracted features.
  # Dropout(0.5) randomly zeroes activations during training,
  # acting as an ensemble regulariser to reduce co-adaptation.
  layer_dense(units = 256, activation = "relu", name = "dense_1") %>%
  layer_dropout(rate = 0.5, name = "dropout") %>%

  # Layer 6 — Output layer
  # Softmax normalises logits into a probability distribution P(class|image).
  # The predicted class is argmax over these 10 probabilities.
  layer_dense(units = 10, activation = "softmax", name = "output")

summary(model)


# ── COMPILE ───────────────────────────────────────────────────────
# Adam is chosen for its adaptive learning rate per parameter, which
# generally outperforms vanilla SGD on vision tasks without requiring
# extensive LR tuning. Categorical cross-entropy is the standard loss
# for one-hot encoded multi-class classification.
model %>% compile(
  optimizer = "adam",
  loss      = "categorical_crossentropy",
  metrics   = c("accuracy")
)


# ── TRAINING ──────────────────────────────────────────────────────
# 10% of training data is held out as a validation set to monitor
# generalisation and detect overfitting per epoch.
history <- model %>% fit(
  x_train, y_train_ohe,
  epochs           = 10,
  batch_size       = 64,
  validation_split = 0.1,
  verbose          = 1
)


# ── TRAINING DIAGNOSTICS ──────────────────────────────────────────
# A growing gap between train and val curves indicates overfitting.
png("training_history_r.png", width = 1200, height = 500, res = 150)
par(mfrow = c(1, 2))

plot(history$metrics$accuracy,     type = "l", col = "steelblue",
     ylim = c(0, 1), xlab = "Epoch", ylab = "Accuracy", main = "Accuracy")
lines(history$metrics$val_accuracy, col = "darkorange")
legend("bottomright", legend = c("Train", "Validation"),
       col = c("steelblue", "darkorange"), lty = 1)

plot(history$metrics$loss,     type = "l", col = "steelblue",
     xlab = "Epoch", ylab = "Loss", main = "Loss")
lines(history$metrics$val_loss, col = "darkorange")
legend("topright", legend = c("Train", "Validation"),
       col = c("steelblue", "darkorange"), lty = 1)

dev.off()
cat("Training curves saved → training_history_r.png\n")


# ── EVALUATION ────────────────────────────────────────────────────
# Evaluates classification performance on the held-out test set.
results <- model %>% evaluate(x_test, y_test_ohe, verbose = 0)
cat(sprintf("\nTest Accuracy : %.4f\n", results["accuracy"]))
cat(sprintf("Test Loss     : %.4f\n",  results["loss"]))


# ── INFERENCE ON SAMPLE IMAGES ────────────────────────────────────
# Runs inference on 4 test images and visualises results.
# Predicted class = argmax of the softmax output vector.
# Title colour: green = correct prediction, red = misclassification.
indices       <- c(1, 2, 3, 4)   # R is 1-indexed
sample_imgs   <- x_test[indices,,,,drop = FALSE]
sample_labels <- y_test[indices]

preds         <- model %>% predict(sample_imgs)
pred_classes  <- apply(preds, 1, which.max) - 1  # convert back to 0-indexed labels

png("sample_predictions_r.png", width = 1400, height = 500, res = 150)
par(mfrow = c(1, 4), mar = c(2, 1, 4, 1))

for (i in seq_along(indices)) {
  img       <- sample_imgs[i,,,1]
  true_name <- class_names[sample_labels[i] + 1]
  pred_name <- class_names[pred_classes[i]  + 1]
  confidence <- round(max(preds[i,]) * 100, 1)
  color      <- ifelse(true_name == pred_name, "darkgreen", "red")

  image(t(apply(img, 2, rev)), col = grey.colors(256),
        axes = FALSE,
        main = paste0("True:  ", true_name,
                      "\nPred:  ", pred_name,
                      "\nConf: ", confidence, "%"),
        col.main = color, cex.main = 0.8)
}

dev.off()
cat("Predictions saved → sample_predictions_r.png\n")


# ── SAVE MODEL ────────────────────────────────────────────────────
# Serialises the trained model weights and architecture to disk.
model$save("fashion_cnn_r.keras")
cat("Model saved → fashion_cnn_r.h5\n")