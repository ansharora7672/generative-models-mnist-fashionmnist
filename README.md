# Generative Models Comparison: Autoencoder vs VAE vs GAN

A comparative study of three generative model families — a vanilla Autoencoder, a Variational Autoencoder, and a GAN — each trained on MNIST, Fashion-MNIST, and a combined 20-class dataset, evaluated on reconstruction quality, latent-space structure, and generative sample quality.

Originally a group project (Ansh, Henil, Kahan) for *PROG74040 — Advanced Topics in AI and ML*. This copy is maintained for portfolio purposes.

## What's inside

- `1_vanilla_autoencoder.ipynb` — an MLP encoder/decoder (784 → 256 → 128 → latent → 128 → 256 → 784) trained to reconstruct images, in PyTorch.
- `2_variational_autoencoder.ipynb` — a VAE with a custom reparameterization `Sampling` layer, trained with a combined reconstruction + KL-divergence loss, in TensorFlow/Keras.
- `3_gan.ipynb` — a Generator/Discriminator adversarial pair (PyTorch) trained for 70 epochs per dataset, generating novel digit/clothing images from random noise.
- All three models are trained on MNIST, Fashion-MNIST, and a combined dataset (with an offset label scheme to keep all 20 classes distinguishable), so the same architectures can be compared across different visual complexity.
- 2D latent-space scatter plots (using a `LATENT_DIM=2` bottleneck) to visualize how each model organizes classes in latent space.

## Tech stack

Python, PyTorch, TensorFlow/Keras, torchvision, Matplotlib, NumPy

## Run it

```bash
pip install torch torchvision tensorflow matplotlib numpy
jupyter notebook 1_vanilla_autoencoder.ipynb   # or 2_/3_
```

MNIST and Fashion-MNIST download automatically on first run. Training checkpoints are not included in this repo — re-run the training cells to regenerate them.
