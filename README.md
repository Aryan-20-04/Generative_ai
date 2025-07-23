# MNIST GAN - PyTorch Implementation

This project implements a simple Generative Adversarial Network (GAN) using PyTorch to generate handwritten digits from the MNIST dataset. The GAN consists of a fully-connected Generator and Discriminator trained adversarially.

## 📌 Features

- Vanilla GAN trained on MNIST
- Configurable via command-line (epochs, batch size, learning rate, etc.)
- Saves model checkpoints every epoch
- Plots:
  - Discriminator and Generator losses
  - Generated image grids every 10 epochs
- CUDA support
- Optimized DataLoader with multiprocessing

---

## 🖼️ Sample Output

Generated samples after training (visualized every 10 epochs):

<p align="center">
  <img src="samples/sample_epoch_10.png" width="300">
</p>

---

## 🔧 Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib

Install via:

```bash
pip install torch torchvision matplotlib
