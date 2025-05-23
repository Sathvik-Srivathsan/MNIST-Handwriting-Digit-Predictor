import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np # Import numpy for argmax and reshaping

# Load MNIST data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model structure
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2), # Added dropout for better generalization
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# This is the key step that was missing.
# Training for a few epochs to get reasonable accuracy.
print("Training the model...")
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
print("Model training complete.")

# Evaluate the model on the test set
print("\nEvaluating model performance on the test set:")
loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Select a test image (e.g., the 2nd image, index 1)
sample_image_index = 1
sample_image = test_images[sample_image_index]
sample_label = test_labels[sample_image_index]

# Predict the label for the selected sample image
# The model expects a batch of images, so we reshape the single image
# and add a batch dimension using np.expand_dims or reshape.
# Using sample_image.reshape(1, 28, 28) is also correct.
image_for_prediction = np.expand_dims(sample_image, axis=0)
prediction = model.predict(image_for_prediction)

# The prediction is an array of probabilities for each class (0-9).
# We need to find the class with the highest probability.
predicted_label = np.argmax(prediction[0])

# Visualize the sample image and the prediction
plt.figure()
plt.imshow(sample_image, cmap='gray')
plt.title(f"True Label: {sample_label}, Predicted: {predicted_label}")
plt.xlabel(f"Confidence: {np.max(prediction[0]*100):.2f}%")
plt.grid(False)
plt.show()

# You can also plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

print("\nTo predict on a different image, change 'sample_image_index'.")
