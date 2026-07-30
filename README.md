# Classification Algorithms

Author: Georgia Kalapotharakou

This repository contains implementations of several foundational machine learning classification algorithms. The project focuses on binary classification tasks using MNIST, CIFAR-10, synthetic blob data, and small interpretable datasets.

## Project Overview

The goal of this project was to implement core classification algorithms, evaluate their performance, and analyze how different models and hyperparameters affect accuracy and generalization.

The project includes:

- A baseline classifier
- K-Nearest Neighbors
- Perceptron
- Decision Tree classifier
- Data preprocessing and accuracy evaluation
- Experimental analysis with plots and written results

## Algorithms Implemented

- **Most Frequent Class Classifier** - predicts the most common label from the training set
- **K-Nearest Neighbors (KNN)** - classifies examples using the majority label among the nearest training examples
- **Perceptron** - learns a linear decision boundary using the perceptron update rule
- **Decision Tree** - recursively splits on features using a classification-error-based heuristic

## Files

- `simple_classifier.py` - Most Frequent Class baseline classifier
- `knn.py` - K-Nearest Neighbors classifier
- `perceptron.py` - Perceptron classifier
- `dt.py` - Decision Tree classifier
- `main.py` - runs the experiments and generates plots/results
- `utils.py` - data processing, accuracy computation, plotting, and dataset helpers
- `datasets.py` - small datasets used for decision tree experiments
- `REPORT.pdf` - written analysis and experimental results
- `requirements.txt` - Python package dependencies

## Datasets

This project uses:

- MNIST digits 3 and 8 for binary digit classification
- CIFAR-10 airplane vs frog classification
- Synthetic blob datasets for perceptron experiments
- Small interpretable datasets for decision tree experiments

## Requirements

Install dependencies with `pip install -r requirements.txt`

## How to Run

Run the main analysis script `python main.py`

This script runs the experiments for the different classifiers.
