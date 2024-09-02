# CIFAR-100 Classification using 3-NN and Various Representation Spaces

## Project Overview

This project implements a machine learning pipeline to classify images from the CIFAR-100 dataset using a 3-Nearest Neighbors (3-NN) classifier. The key focus of the project is to explore how different data representation spaces affect classification performance. The representations used include:

- **Original Pixel Values**
- **Autoencoder Features**
- **CLIP Embeddings**
- **ResNet Embeddings**

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Representations](#representations)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [License](#license)

## Dataset

The project uses the CIFAR-100 dataset, which consists of 60,000 32x32 color images in 100 classes, with 600 images per class. The dataset is split into 50,000 training images and 10,000 test images.

## Representations

The project explores the impact of different representation spaces on the performance of the 3-NN classifier:

1. **Original Features:**
   - Uses the raw pixel values of the images.

2. **Autoencoder Features:**
   - A neural network-based autoencoder is trained to compress the images into a lower-dimensional representation and then reconstruct them. The compressed (bottleneck) features are used for classification.

3. **CLIP Embeddings:**
   - CLIP (Contrastive Language–Image Pretraining) embeddings are generated using a pre-trained CLIP model. These embeddings map images and text into a shared semantic space.

4. **ResNet Embeddings:**
   - A pre-trained ResNet50 model is used to extract high-level feature embeddings from the images, capturing complex visual patterns.

## Results

The classification accuracy achieved with each representation space is as follows:

- **Original Features:** 42.3%
- **Autoencoder Features:** 50.3%
- **CLIP Embeddings:** 48.5%
- **ResNet Embeddings:** 49.4%

The autoencoder features provided the highest accuracy, suggesting that task-specific learning via the autoencoder was most effective for this dataset and classifier combination.

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/cifar100-classification.git
   cd cifar100-classification
    Install Dependencies: Ensure you have all necessary libraries installed. See the Dependencies section for details.

    Run the Notebook: Open and execute the provided Jupyter notebook (classification_pipeline.ipynb) in Google Colab or your local environment.

    Modify Class Selection: To test different subsets of 10 classes, modify the selected_classes list in the notebook.

Dependencies

    Python 3.x
    Numpy
    TensorFlow
    PyTorch
    TorchVision
    OpenAI's CLIP
    Scikit-learn
    Matplotlib

You can install the required dependencies via pip: pip install numpy tensorflow torch torchvision git+https://github.com/openai/CLIP.git scikit-learn matplotlib
License

This project is licensed under the MIT License - see the LICENSE file for details.

License

This project is licensed under the MIT License - see the LICENSE file for details.


### Instructions for Use:
- Replace `"https://github.com/your-username/cifar100-classification.git"` with your actual GitHub repository URL.
- Add the `LICENSE` file if you plan to use the MIT License, or adjust the license section according to your chosen license.

This `README.md` provides a clear overview of the project, instructions on how to run it, and relevant details about dependencies and results. It should make your repository easy to understand and use for others who might be interested in your work.
