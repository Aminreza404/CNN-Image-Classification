# CNN Image Classification with PyTorch

A simple Convolutional Neural Network (CNN) for image classification on the **CIFAR-10 dataset**, implemented using **PyTorch**.

This project demonstrates the complete workflow of an image classification task, including dataset preparation, data augmentation, CNN model design, training, evaluation, and visualization of loss and accuracy.

## Overview

The model is trained to classify images into 10 different categories from the CIFAR-10 dataset:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

The project uses a CNN consisting of three convolutional layers followed by fully connected layers.

## Dataset

This project uses the **CIFAR-10** dataset.

CIFAR-10 contains:

* **50,000** training images
* **10,000** test images
* RGB images with a resolution of **32 × 32 pixels**
* **10** image classes

The dataset is automatically downloaded using `torchvision.datasets.CIFAR10`.

## Data Preprocessing

The training and test datasets are transformed using PyTorch's `torchvision.transforms`.

The preprocessing pipeline includes:

```python
transforms.RandomHorizontalFlip()
transforms.RandomCrop(32, padding=4)
transforms.ToTensor()
transforms.Normalize((0.5, 0.5, 0.5),
                     (0.5, 0.5, 0.5))
```

### Data Augmentation

Two augmentation techniques are used:

* **Random Horizontal Flip** — randomly flips images horizontally.
* **Random Crop** — crops a 32 × 32 region after adding padding.

These transformations introduce variation into the training data and can help the model generalize better.

## CNN Architecture

The neural network consists of three convolutional layers:

```text
Input: 3 × 32 × 32

        ↓
Conv2D: 3 → 32
ReLU
Max Pooling

        ↓
Conv2D: 32 → 64
ReLU
Max Pooling

        ↓
Conv2D: 64 → 128
ReLU
Max Pooling

        ↓
Flatten

        ↓
Fully Connected: 128 × 4 × 4 → 512
ReLU
Dropout (p = 0.5)

        ↓
Fully Connected: 512 → 10

        ↓
Output: 10 classes
```

The final layer produces 10 outputs corresponding to the 10 CIFAR-10 classes.

## Training

The model is trained using:

* **Loss function:** Cross-Entropy Loss
* **Optimizer:** Adam
* **Learning rate:** `0.001`
* **Weight decay:** `1e-4`
* **Dropout:** `0.5`
* **Batch size:** `32`
* **Number of epochs:** `10`

The code automatically detects whether CUDA is available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Therefore, training can use a CUDA-enabled GPU when available, otherwise it runs on the CPU.

## Evaluation

During training, the following quantities are recorded for every epoch:

* Training loss
* Test loss
* Training accuracy
* Test accuracy

The results are then visualized using Matplotlib.

Two plots are generated:

1. **Training and test loss vs. epoch**
2. **Training and test accuracy vs. epoch**

These plots help evaluate the learning process and identify potential overfitting or underfitting.

## Requirements

Install the required Python packages with:

```bash
pip install numpy matplotlib torch torchvision tqdm
```

A CUDA-enabled installation of PyTorch is recommended if you want to train the model on an NVIDIA GPU.

## How to Run

Clone the repository:

```bash
git clone https://github.com/Aminreza404/CNN-Image-Classification.git
```

Navigate to the project directory:

```bash
cd CNN-Image-Classification
```

Run the Python script:

```bash
python your_script.py
```

The CIFAR-10 dataset will be downloaded automatically the first time the program is executed.

## Project Workflow

```text
CIFAR-10 Dataset
       ↓
Data Augmentation
       ↓
Data Normalization
       ↓
CNN Model
       ↓
Training with Adam
       ↓
Model Evaluation
       ↓
Loss & Accuracy Visualization
```

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* tqdm

## Future Improvements

Possible improvements to this project include:

* Adding Batch Normalization
* Using learning-rate scheduling
* Training for more epochs
* Comparing different CNN architectures
* Adding validation-set evaluation
* Saving and loading trained model weights
* Generating a confusion matrix
* Evaluating precision, recall, and F1-score
* Testing more advanced architectures such as ResNet

## Author

**Aminreza Safarpour**

GitHub: [Aminreza404](https://github.com/Aminreza404)

