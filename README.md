# Fire_classification

This project implements an image-based fire detection system using a lightweight convolutional neural network (CNN). The architecture includes two convolutional layers followed by max pooling, and two fully connected layers with a sigmoid activation at the output for binary classification.

The model is trained on the Fire Dataset from Kaggle: Fire Dataset on Kaggle, which contains labeled images of fire and non-fire scenes. The dataset is divided into training, validation, and test sets with the following class distribution:

Not-fire (class 0): 172 (train), 34 (val), 37 (test)

Fire (class 1): 526 (train), 115 (val), 114 (test)

After training for 10 epochs, the model reached its best performance at epoch 6, achieving:

Training Accuracy: 95.56%

Validation Accuracy: 96.64%

Test Accuracy: 98.01%

Precision: 98.26%

Recall: 99.12%

F1-score: 98.69%

This project demonstrates that even a simple CNN architecture can effectively distinguish between fire and non-fire images with high accuracy and recall, making it suitable for early-stage fire detection systems.
