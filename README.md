MNIST Data Handling: Loads and prepares the famous MNIST dataset of handwritten digits.

Simple Neural Network: Creates a basic deep learning model with:

    An input layer to flatten images.
    A hidden layer with 128 neurons (ReLU activation).
    A dropout layer to prevent overfitting.
    An output layer with 10 neurons (softmax activation) for digit classification.

Training & Evaluation: Compiles and trains the model for 5 epochs, then evaluates its accuracy on unseen data.

Prediction & Visualization: Predicts a digit from the test set and visually displays the image, true label, and the model's prediction with confidence.

Performance Tracking: Plots training and validation accuracy/loss over time to show model learning.
