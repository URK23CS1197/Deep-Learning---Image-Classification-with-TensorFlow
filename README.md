# Deep-Learning---Image-Classification-with-TensorFlow
Image classification model using TensorFlow and the Fashion MNIST dataset. The model is trained to recognize clothing categories with a simple neural network and visualizes training and validation accuracy to evaluate performance.

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: NIRANSON CDK

*INTERN ID* : CT04DG534

*DOMAIN*: DATA SCIENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

In this task, I implemented a deep learning model to solve an image classification problem using TensorFlow, a widely used open-source deep learning framework developed by Google. The project focused on classifying fashion images using the Fashion MNIST dataset, which is a modern, more challenging alternative to the traditional handwritten digits dataset (MNIST). It contains grayscale images of 70,000 fashion products from 10 different categories, such as t-shirts, trousers, sandals, sneakers, etc.

Data Loading and Preprocessing
The dataset was loaded using the built-in tensorflow.keras.datasets.fashion_mnist module. It comes pre-split into training and test datasets: 60,000 images for training and 10,000 for testing.

Each image in the dataset is a 28x28 grayscale image, with pixel values ranging from 0 to 255. I normalized the image data by dividing all pixel values by 255.0 to bring them into the range [0, 1], which speeds up training and improves model convergence.

Model Architecture
I used a Sequential model from Keras, which allows stacking layers one by one. The model architecture included:

A Flatten layer to convert the 2D image (28x28) into a 1D array of 784 pixels.

A Dense (fully connected) layer with 128 neurons and ReLU activation to learn non-linear patterns.

A Dropout layer with a rate of 0.2 to prevent overfitting by randomly dropping 20% of the nodes during each training step.

An output Dense layer with 10 neurons, using softmax activation to produce a probability distribution over the 10 fashion categories.

Model Compilation and Training
I compiled the model using:

Optimizer: Adam – an adaptive learning rate optimizer well-suited for deep learning.

Loss Function: Sparse Categorical Crossentropy – suitable for multiclass classification with integer labels.

Metrics: Accuracy – to track classification performance.

The model was then trained for 5 epochs on the training data, with validation on the test set. During training, both the training and validation accuracy and loss were tracked.

Model Evaluation and Visualization
After training, I evaluated the model on the test dataset and achieved a respectable accuracy, demonstrating that the model was able to generalize well to unseen fashion images.

To visualize model performance, I plotted graphs of training and validation accuracy and loss across epochs using Matplotlib. This helped assess whether the model was underfitting or overfitting and provided valuable insights into the learning dynamics.

*OUTPUT*

The final result was a functional deep learning image classification model that can correctly predict fashion categories with high accuracy. The model was designed to be lightweight, fast, and easy to adapt to other similar image datasets.This task gave me hands-on experience with the core concepts of deep learning, including building neural networks, working with image data, training models, preventing overfitting with dropout, and interpreting training curves. It served as a strong foundation for more advanced deep learning tasks like CNNs and transfer learning.


![Image](https://github.com/user-attachments/assets/be01c47a-e6a7-4502-8b16-ca888a9ebd4d)

![Image](https://github.com/user-attachments/assets/92566108-8b24-4f24-81d8-d6e676ae9497)
