<div align="center">
  <h1>🔮 Laplace Redux: Effortless Bayesian Deep Learning</h1>
  <p><i>Bringing principled, efficient, and practical uncertainty quantification to deep learning models.</i></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python" />
    <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch" />
    <img src="https://img.shields.io/badge/AI-Bayesian_Deep_Learning-green.svg" alt="Bayesian DL" />
    <img src="https://img.shields.io/badge/Library-Laplace-orange.svg" alt="Laplace" />
  </p>
</div>

---

## 🚀 Overview

Deep learning models are powerful, but they often struggle to say "I don't know." **Laplace Redux** bridges this gap. Based on the paper *"Laplace Redux – Effortless Bayesian Deep Learning"* (Daxberger et al., 2021), this project successfully reproduces and extends the application of the Laplace Approximation (LA) to bring principled uncertainty quantification to neural networks.

> 💡 **The Core Engineering:** *We engineered a robust pipeline using PyTorch and the `laplace` library to transform standard deterministically trained models into Bayesian ones. By fitting a Gaussian distribution around the Maximum A Posteriori (MAP) estimate, we enabled efficient uncertainty quantification that rivals deep ensembles at a fraction of the computational cost.*

## ⚙️ How It Works

Applying Bayesian principles to deep learning is traditionally computationally heavy. This project demonstrates how the **Laplace Approximation** acts as an efficient, post-hoc technique:

* 🧠 **Post-Hoc Bayesian Treatment:** The system takes a standard, pre-trained neural network (the MAP estimate) and retroactively fits a Gaussian distribution around its weights to capture uncertainty.
* 🔀 **Versatile LA Variants:** We explored and implemented multiple structural variants of the Laplace Approximation to balance cost and accuracy, including:
  * **Full Laplace:** The direct, comprehensive approach.
  * **Subspace Laplace:** Projecting into lower-dimensional spaces for efficiency.
  * **Swag Laplace:** Utilizing Stochastic Weight Averaging-Gaussian techniques.
* 🎯 **Hyperparameter Sensitivity Analysis:** We conducted extensive testing on the interplay between link approximations, Hessian structures, and regularization levels, proving that there is no "one-size-fits-all" approach, and providing actionable insights for practitioners.
* 🛡️ **Out-of-Distribution (OOD) Robustness:** We demonstrated how proper hyperparameter tuning and uncertainty quantification drastically improve a model's ability to handle anomalous or out-of-distribution data.

## 🛠️ Tech Stack

This project is built on a modern deep learning and probabilistic modeling stack:

* 🤖 **Deep Learning:** `PyTorch` | `Torchvision`
* 📊 **Probabilistic AI:** `laplace` (PyTorch library) | `Bayesian Neural Networks`
* ⚙️ **Analysis & Metrics:** `NumPy` | `Pandas` | `Matplotlib`
* 🧪 **Techniques:** `MAP Estimation` | `Hessian Matrix Approximations` | `Uncertainty Quantification`

---

## Blog
The complete blog can be found [here](BLOG.md)

## How to Run

1. Make sure you have Docker and Docker Compose installed
2. Run the setup script: `./setup_docker.sh`
3. Download models from [Google Drive](https://drive.google.com/file/d/17cI3dhconEwLj5J3XTEgbPlzYTUoSxAk/view?usp=share_link)
4. Extract them to the `models` directory
5. Build and start the container: `docker-compose up -d`
6. Run experiments:
   - `docker-compose exec laplace ./tests/run_uq_baselines.sh`
   - `docker-compose exec laplace ./tests/run_uq_laplace.sh`
   - `docker-compose exec laplace ./tests/run_uq_subspace.sh`
