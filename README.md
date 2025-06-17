# Building a Neural Network from Scratch to Classify MNIST and Fashion-MNIST Datasets

Welcome to my neural network implementation project! This notebook walks through the process of building, training, and evaluating a feedforward neural network from scratch using only NumPy.

## 📂 How to Explore This Project

To explore the code and replicate the experiments:

1. Clone or download this repository.
2. Open the `exploration.ipynb` notebook in Jupyter or any compatible environment (e.g., VS Code, Google Colab).
3. Make sure you have the required packages:
   ```bash
   pip install numpy matplotlib
   ```
4. Run each cell sequentially. For the Fashion-MNIST portion, make sure to manually download the `.gz` files listed in the notebook and place them into a folder named `fashionMNIST`.

---

## 💡 Why Build a Neural Network from Scratch?

In both academic and professional settings, I’ve worked extensively with pre-built models from high-level libraries like TensorFlow and PyTorch. While these tools are efficient, they often abstract away the inner workings of the models—especially when it comes to fine-tuning or optimization.

This project is my effort to deepen my understanding of how neural networks really work under the hood. I intentionally avoided machine learning libraries and instead relied solely on `NumPy` to implement everything—from weight initialization and forward propagation to backpropagation and gradient updates.

---

## 🔧 What the Code Does

The notebook is divided into clear, functional sections:

### 1. Neural Network Implementation

- Created a class `NeuralNetwork` to build a fully connected network with ReLU activations in hidden layers and Softmax in the output layer.
- Includes forward propagation, backpropagation, and a manual parameter update method using gradient descent.

### 2. Custom Loss Function

- Implemented Categorical Cross-Entropy Loss with numerical stability (using `np.clip`) to avoid issues like log(0).

### 3. Data Handling

- Loaded and normalized the MNIST dataset using basic NumPy functions.
- Reshaped the images and labels into appropriate formats for training.

### 4. Training Loop

- Trained the network over 2 epochs with a batch size of 32.
- Tracked accuracy and loss during training for visualization.
- Implemented manual batching, gradient updates, and loss calculation from scratch.

### 5. Evaluation

- Evaluated the trained model on the MNIST test set, achieving high accuracy.
- Visualized both correct and incorrect predictions to interpret performance.
- Plotted loss and accuracy curves over training steps.

### 6. Fashion-MNIST Extension

- Downloaded and preprocessed the Fashion-MNIST dataset.
- Trained a new neural network model using the same architecture and methods.
- Evaluated its performance and confirmed generalization capability across a different dataset.

---

## 🧠 Challenges & Solutions

- **Gradient Stability**: Early in training, I encountered instability due to improper learning rate and weight initialization. I solved this by adjusting initialization scale and introducing learning rate decay.
- **Backpropagation Bugs**: Implementing backwards propagation manually revealed a few subtle bugs, especially with ReLU gradients. Debugging involved comparing layer outputs across batches and validating gradients numerically.
- **Shape Mismatches**: Tracking the shape of matrices was critical. Errors in reshaping or broadcasting were common and fixed through consistent dimensionality checks.

---

## 📈 Model Performance

### MNIST Digits:

- **Accuracy**: The model consistently achieved over **90% accuracy** on the MNIST test set after just two epochs.
- **Insights**: Misclassifications mostly occurred between visually similar digits like 4 and 9, or 3 and 5.

### Fashion-MNIST:

- **Accuracy**: The model achieved solid performance (~85% accuracy) despite the added complexity and similarity between classes.
- **Insights**: This suggests strong generalization despite no architectural changes.

---

## 🚀 Next Steps

- **Add Dropout & Batch Normalization**: To improve regularization and reduce overfitting.
- **Implement Optimizers**: Such as Adam or RMSProp to speed up convergence and improve accuracy.
- **Train Deeper Architectures**: Try deeper networks with more layers and analyze training dynamics.
- **Visualization Tools**: Add confusion matrix and t-SNE/UMAP for high-dimensional analysis.

---

## 🤝 Acknowledgements

This project was inspired by the desire to demystify neural networks and reinforce foundational concepts in deep learning. I was heavily inspired by Sentdex's 'Neural Networks from Scratch' book, which you can find at https://nnfs.io. I hope it’s as helpful for you to read as it was for me to build.

---
