from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from collections import deque
import warnings
from tqdm import tqdm

# Assumes baselaplace.py is in the same directory or python path
from laplace.baselaplace import DiagLaplace

class SWAGLaplace(DiagLaplace):
    """
    SWAG-as-a-Posterior, a method for uncertainty estimation in deep learning.

    This method uses Stochastic Weight Averaging (SWA) to find a good posterior
    mean and to estimate the covariance of the posterior distribution. The posterior
    is approximated as a Gaussian with a low-rank plus diagonal covariance structure.

    The covariance is estimated as:
    Covariance = Diag(sigma_diag^2) + (1/K) * D @ D.T
    where:
    - sigma_diag^2 is the diagonal variance estimated from the moments of SWA iterates.
    - D is the deviation matrix, where columns are (theta_i - theta_mean).
    - K is the number of models collected (the rank of the low-rank part).

    This class inherits from `DiagLaplace` to reuse its infrastructure for the
    diagonal part of the posterior and overrides methods to include the low-rank component.

    Parameters
    ----------
    model : nn.Module
        The neural network to be trained.
    likelihood : str, {'classification', 'regression'}
        The likelihood of the model.
    swa_start : int, default=10
        The training epoch to start the SWA procedure.
    swa_lr : float, default=0.01
        The constant learning rate to use during SWA.
    swa_freq : int, default=1
        The frequency in epochs to collect model weights for SWA.
    rank : int, default=20
        The maximum number of model weights to store for the low-rank approximation.
        This is the rank 'K' of the low-rank covariance component.
    sigma_noise : float, default=1.0
        Observation noise for regression.
    prior_precision : float, default=1.0
        Prior precision (weight decay).
    temperature : float, default=1.0
        Temperature for the likelihood.
    device : torch.device, optional
        The device to run the computations on.
    """

    # Unique key for this Laplace type
    _key = ("all", "swag_laplace")

    def __init__(
        self,
        model: nn.Module,
        likelihood: str,
        swa_start: int = 10,
        swa_lr: float = 0.01,
        swa_freq: int = 1,
        rank: int = 20,
        sigma_noise: float = 1.0,
        prior_precision: float = 1.0,
        prior_mean: float = 0.0,
        temperature: float = 1.0,
        device=None,
        **kwargs,
    ):
        super().__init__(
            model,
            likelihood,
            sigma_noise=sigma_noise,
            prior_precision=prior_precision,
            prior_mean=prior_mean,
            temperature=temperature,
            **kwargs,
        )

        if device is not None:
            self.device = device
        else:
            self.device = next(model.parameters()).device
        self.model.to(self.device)
        
        # SWA hyperparameters
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.swa_freq = swa_freq
        
        # Rank of the low-rank covariance approximation
        self.rank = rank

        # Storage for SWA statistics
        self._collected_weights = deque(maxlen=self.rank)
        self.n_models_collected = 0
        self.swag_mean_vec = None
        self.swag_sq_mean_vec = None

        # Low-rank deviation matrix D
        self.D = None

    def fit(self,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            epochs: int,
            progress_bar: bool = False,
            ):
        """
        Train the model using the SWAG procedure and compute the posterior approximation.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training data.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training.
        criterion : nn.Module
            The loss function.
        epochs : int
            The total number of epochs to train for.
        progress_bar : bool, default=False
            Whether to display a progress bar during training.
        """
        if self.swa_start >= epochs:
            warnings.warn("SWA start epoch is after the total number of epochs. SWA will not run.")

        self.n_data = len(train_loader.dataset)
        self.model.train()
        
        iterator = range(epochs)
        if progress_bar:
            iterator = tqdm(iterator, desc="SWAG Training")

        for epoch in iterator:
            # SWA learning rate schedule
            is_swa_epoch = epoch >= self.swa_start
            if is_swa_epoch:
                # Set to constant SWA learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.swa_lr

            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if progress_bar:
                iterator.set_postfix(loss=f"{avg_loss:.4f}", swa=is_swa_epoch)
            
            # Collect weights during SWA phase
            if is_swa_epoch and (epoch - self.swa_start) % self.swa_freq == 0:
                self._update_swag_stats()

        # After training, compute the final SWAG posterior
        self._compute_swag_posterior()
        
        # Store final loss (average loss of the last epoch)
        self.loss = avg_loss

    def _update_swag_stats(self):
        """Update running averages and collect weights for the low-rank part."""
        current_params = parameters_to_vector(self.model.parameters()).detach().clone()
        
        # Update running moments for diagonal variance
        if self.swag_mean_vec is None:
            self.swag_mean_vec = torch.zeros_like(current_params)
            self.swag_sq_mean_vec = torch.zeros_like(current_params)

        n = self.n_models_collected + 1
        # Update first moment (mean)
        self.swag_mean_vec = (self.swag_mean_vec * (n - 1) + current_params) / n
        # Update second moment (for variance)
        self.swag_sq_mean_vec = (self.swag_sq_mean_vec * (n - 1) + current_params**2) / n
        
        # Store weights for low-rank approximation
        self._collected_weights.append(current_params)
        self.n_models_collected += 1
        
    def _compute_swag_posterior(self):
        """Compute the final posterior from the collected SWAG statistics."""
        if self.n_models_collected == 0:
            raise RuntimeError("SWAG training did not collect any models. Check `swa_start` and `epochs`.")

        # 1. Set the posterior mean
        self.mean = self.swag_mean_vec.clone()
        vector_to_parameters(self.mean, self.model.parameters()) # Set model to mean
        
        # 2. Compute the diagonal precision H
        diag_var = self.swag_sq_mean_vec - self.swag_mean_vec**2
        # Clamp for numerical stability
        diag_var = torch.clamp(diag_var, min=1e-6)
        
        # The diagonal part of the Hessian is the inverse of the variance
        self._init_H() # H is a flat vector for DiagLaplace
        self.H = 1.0 / diag_var

        # THE FOLLOWING LINES SHOULD BE REMOVED:
        # if hasattr(self, 'prior_precision'):
        #     self.H += self.prior_precision_diag

        # 3. Construct the low-rank deviation matrix D
        if self.rank > 0 and len(self._collected_weights) > 0:
            # D has shape (n_params, rank)
            self.D = torch.stack(
                [w - self.mean for w in self._collected_weights], dim=1
            ).to(self.device)
            # Adjust rank if fewer models were collected than requested
            self.rank = self.D.shape[1]
        else:
            self.D = None
            self.rank = 0

    def sample(self, n_samples: int = 100):
        """
        Sample from the SWAG posterior N(mean, Cov).

        Cov = Diag(1/H) + (1/rank) * D @ D.T
        """
        # Get samples from the diagonal part using the parent method
        # This samples from N(mean, Diag(1/H))
        samples = super().sample(n_samples)

        # Add samples from the low-rank part
        if self.D is not None and self.rank > 0:
            # Sample from standard normal N(0, I)
            z = torch.randn(self.rank, n_samples, device=self.device)
            
            # Low-rank samples: (D @ z) / sqrt(rank)
            # Shape of D is (n_params, rank), z is (rank, n_samples)
            # Resulting shape is (n_params, n_samples)
            low_rank_samples = (self.D @ z) / torch.sqrt(torch.tensor(self.rank))
            
            # Add to diagonal samples
            # Transpose to (n_samples, n_params) to match super().sample() output
            samples += low_rank_samples.T

        return samples

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        """
        Compute the predictive variance, Var[f*] = J Cov J.T
        Var[f*] = J (Diag(1/H) + (1/rank) * D @ D.T) J.T
                 = J Diag(1/H) J.T + (1/rank) * (J @ D) @ (J @ D).T
        """
        # Get predictive variance from the diagonal part from the parent class
        f_var_diag = super().functional_variance(Js)

        if self.D is not None and self.rank > 0:
            # Js has shape (batch, outputs, params)
            # D has shape (params, rank)
            # JD has shape (batch, outputs, rank)
            JD = torch.einsum('bop,pk->bok', Js, self.D)
            
            # (JD) @ (JD).T -> (batch, outputs, rank) @ (batch, rank, outputs)
            # This computes the low-rank variance component
            f_var_low_rank = torch.einsum('bok,brk->bor', JD, JD) / self.rank
            
            return f_var_diag + f_var_low_rank
        else:
            return f_var_diag
