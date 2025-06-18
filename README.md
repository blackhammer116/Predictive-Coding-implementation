# Predictive Coding Implementation

This repository contains an implementation of a **Predictive Coding Network (PCN)**, a biologically inspired framework for neural computation. Predictive coding is a theory of brain function that suggests the brain continuously generates predictions about sensory input and updates its internal model based on prediction errors.

The implementation is designed to work with the MNIST dataset for handwritten digit classification, demonstrating how predictive coding can be applied to machine learning tasks.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Mathematical Background](#mathematical-background)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)


---

## Overview

Predictive coding is a computational framework where the system minimizes prediction errors at each layer of a hierarchical model. This implementation uses a two-layer predictive coding network to classify MNIST digits. The network learns by iteratively updating its latent variables and weights to minimize prediction errors.

---

## Features

- **Biologically Inspired Learning**: Implements predictive coding principles for neural computation.
- **MNIST Classification**: Trains and tests the network on the MNIST dataset.
- **Customizable Architecture**: Easily modify the number of layers, neurons, and learning rates.
- **Error Propagation**: Uses prediction errors to update latent variables and weights.
- **Efficient Training**: Supports batch processing and GPU acceleration.

---

## Mathematical Background

Predictive coding minimizes a free energy function or prediction error, defined as:

E= 
2
1
​
 ∥ε 
1
​
 ∥ 
2
 + 
2
1
​
 ∥ε 
2
​
 ∥ 
2

Where:
- ε_1 = x_1 - f(x_0) : Prediction error at the first layer.
- ε_2 = x_2 - g(x_1) : Prediction error at the second layer.

The updates for the latent variables (\( x_1, x_2 \)) are derived as:

\[
\Delta x_1 = \epsilon_1 - W_2^T \epsilon_2 \cdot f'(x_1)
\]
\[
\Delta x_2 = \epsilon_2
\]

The weights (\( W_1, W_2 \)) are updated using gradient descent:

\[
\Delta W_1 = -\eta \frac{\partial E}{\partial W_1}, \quad \Delta W_2 = -\eta \frac{\partial E}{\partial W_2}
\]

---

## Project Structure


---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Predictive-Coding-implementation.git
   cd Predictive-Coding-implementation

2. Install dependencies:
```bash pip install -r requirements.txt```

## Usage
Both training and testing is done sequencially, run the program and see the results
``` python main ```

## Customization
You can customize the network architecture and training parameters by modifying the PCN class in src/model/pcn.py or the train_pcn function in src/main.py.

**Key Parameters** :
- ```input_dim```: Number of input features (default: 784 for MNIST).
- ```hidden_dim```: Number of neurons in the hidden layer (default: 256).
- ```output_dim```: Number of output classes (default: 10 for MNIST).
- ```n_inference_steps```: Number of inference steps per forward pass.
- ```lr_infer```: Learning rate for latent variable updates.
- ```lr_weight```: Learning rate for weight updates.