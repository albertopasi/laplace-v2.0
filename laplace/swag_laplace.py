from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from laplace.baselaplace import DiagLaplace
from laplace.utils.swag import SWAG
import gc
from laplace.utils.enums import Likelihood, LinkApprox, PredType


class SWAGLaplace(DiagLaplace):
    """Laplace approximation using SWAG covariance as the Hessian approximation.
    
    This combines Stochastic Weight Averaging Gaussian (SWAG) with Laplace approximation,
    using SWAG's parameter distribution to approximate the posterior precision matrix.
    Unlike standard DiagLaplace, this implementation also supports low-rank corrections
    from SWAG's covariance estimation for more accurate uncertainty.
    """

    _key = ("all", "swag_laplace")

    def __init__(
        self,
        model: nn.Module,
        likelihood: str,
        n_models: int = 20,
        start_epoch: int = 0,
        swa_freq: int = 1,
        swa_lr: float = 0.05,
        max_num_models: int = 20,
        var_clamp: float = 1e-6,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        device=None,
        **kwargs
    ):
        super().__init__(
            model, 
            likelihood, 
            sigma_noise=sigma_noise,
            prior_precision=prior_precision, 
            prior_mean=prior_mean,
            temperature=temperature, 
            **kwargs
        )

        self.device = device if device is not None else next(model.parameters()).device
        
        # Initialize SWAG
        self.swag = SWAG(
            model=model,
            n_models=n_models,
            start_epoch=start_epoch,
            swa_freq=swa_freq,
            swa_lr=swa_lr,
            max_num_models=max_num_models,
            var_clamp=var_clamp,
            device=self.device
        )
        
        # Initialize storage for SWAG statistics
        self._init_swag_storage()

    def _init_swag_storage(self):
        """Initialize storage for SWAG mean and covariance components"""
        self.swag_mean = None
        self.swag_covariance = None
        self.U_full = None
        self.S_full = None

    def fit(self, 
            train_loader: DataLoader,
            override: bool = True,
            progress_bar: bool = False,
            optimizer: torch.optim.Optimizer | None = None,
            criterion: nn.Module | None = None,
            epochs: int | None = None,
            start_epoch: int = 0,
            **kwargs
        ):
        """Fit the SWAG-Laplace approximation using SWAG training.
        
        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            DataLoader for training data
        override : bool, default=True
            Whether to override previous fit
        progress_bar : bool, default=False
            Display progress bar during training
        optimizer : torch.optim.Optimizer, required
            Optimizer for SWAG training
        criterion : nn.Module, required
            Loss function for SWAG training
        epochs : int, required
            Number of epochs for SWAG training
        start_epoch : int, default=0
            Starting epoch for SWAG training
        """
        # Extract parameters from kwargs if they were passed that way,
        # otherwise use the explicitly passed ones.
        opt = optimizer if optimizer is not None else kwargs.pop('optimizer', None)
        crit = criterion if criterion is not None else kwargs.pop('criterion', None)
        eps = epochs if epochs is not None else kwargs.pop('epochs', None)

        if opt is not None and crit is not None and eps is not None:
            self.train_swag(
                train_loader=train_loader,
                optimizer=opt,
                criterion=crit,
                epochs=eps,
                start_epoch=start_epoch,
                progress_bar=progress_bar,
                **kwargs
            )
        else:
            raise ValueError(
                "SWAGLaplace.fit requires 'optimizer', 'criterion', and 'epochs' "
                "to be provided for SWAG training."
            )
            
        # Save dataset size for scaling
        self.n_data = len(train_loader.dataset)
    
    def train_swag(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        start_epoch: int = 0,
        progress_bar: bool = False,
        **kwargs
    ):
        """Train model using SWAG and extract statistics for Laplace approximation.
        
        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            DataLoader for training data
        optimizer : torch.optim.Optimizer
            Optimizer for SWAG training
        criterion : nn.Module 
            Loss function for SWAG training
        epochs : int
            Number of epochs for SWAG training
        start_epoch : int, default=0
            Starting epoch for SWAG training
        progress_bar : bool, default=False
            Display progress bar during training
        """
        # Train the model using SWAG
        self.swag.fit(
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            start_epoch=start_epoch,
            progress_bar=progress_bar
        )
        
        # Get SWAG statistics
        var, (U, S) = self.swag.get_covariance()
        
        # Store SWAG mean (on CPU to save GPU memory)
        self.swag_mean = [mean.clone().cpu() for mean in self.swag.mean]
        
        # Store SWAG covariance (on CPU to save GPU memory)
        self.swag_covariance = {
            'var': [v.cpu() for v in var],
            'U': [u.cpu() if u is not None else None for u in U],
            'S': [s.cpu() if s is not None else None for s in S]
        }
        
        # Clear SWAG's internal storage to free memory
        if hasattr(self.swag, '_mean_list'):
            del self.swag._mean_list
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Compute Laplace approximation using SWAG statistics
        self._compute_laplace_approximation()
        
        # Get loss value for marginal likelihood computation
        # This is approximate since SWAG doesn't directly provide this
        with torch.no_grad():
            total_loss = 0.0
            n_batches = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                n_batches += 1
        
        # Store loss value
        self.loss = total_loss / n_batches

    def _compute_laplace_approximation(self):
        # Set the model parameters to SWAG mean
        with torch.no_grad():
            for model_param, mean_val in zip(self.model.parameters(), self.swag_mean):
                model_param.data.copy_(mean_val.to(self.device))
        
        # Update self.mean
        self.mean = parameters_to_vector(self.model.parameters()).detach()

        # Initialize Hessian diagonal
        self._init_H()
        
        # Extract SWAG components
        var_list = self.swag_covariance['var']
        U_list = self.swag_covariance['U'] 
        S_list = self.swag_covariance['S']
        
        # Convert diagonal variance to precision
        eps = 1e-6  # Small value for numerical stability
        var_flattened = torch.cat([v.reshape(-1) for v in var_list]).to(self._device).to(self._dtype)
        var_flattened = torch.clamp(var_flattened, min=eps)
        
        # Set diagonal precision from SWAG variance
        self.H = 1.0 / var_flattened
        
        # Process and store low-rank components if available
        self.U_full = None
        self.S_full = None
        
        if any(u is not None for u in U_list) and any(s is not None for s in S_list):
            # Get ranks of each U matrix, handling different dimensions
            ranks = []
            for u in U_list:
                if u is not None:
                    # If u has at least 2 dimensions, use shape[1], otherwise use 1
                    if u.dim() >= 2:
                        ranks.append(u.shape[1])
                    else:
                        # For 1D tensors, treat as rank 1
                        ranks.append(1)
            
            # If we have any ranks, get the maximum
            if ranks:
                max_rank = max(ranks)
                n_params = len(var_flattened)
                
                # Initialize full U matrix with zeros
                U_full = torch.zeros((n_params, max_rank), device=self._device, dtype=self._dtype)
                S_full = torch.zeros(max_rank, device=self._device, dtype=self._dtype)
                
                # Track parameter offset for correct placement
                param_offset = 0
                
                # Fill in the low-rank components
                for U_mat, S_vec, var_vec in zip(U_list, S_list, var_list):
                    if U_mat is not None and S_vec is not None:
                        param_size = var_vec.numel()
                        
                        # Handle U matrices with different dimensions
                        if U_mat.dim() >= 2:
                            rank = U_mat.shape[1]
                            U_reshaped = U_mat.reshape(param_size, rank)
                        else:
                            rank = 1
                            U_reshaped = U_mat.reshape(param_size, 1)
                        
                        # Handle S vectors with different dimensions
                        if S_vec.dim() == 0:  # scalar
                            S_reshaped = S_vec.unsqueeze(0)
                        else:
                            S_reshaped = S_vec
                        
                        # Copy the low-rank component to the right location
                        U_full[param_offset:param_offset + param_size, :rank] = U_reshaped.to(self._device).to(self._dtype)
                        S_full[:rank] = S_reshaped.to(self._device).to(self._dtype)
                        
                        param_offset += param_size
                
                # Store for efficient posterior sampling
                self.U_full = U_full
                self.S_full = S_full
        
        # Add prior precision
        if hasattr(self, 'prior_precision'):
            if isinstance(self.prior_precision, float):
                self.H += self.prior_precision
            else:
                self.H += self.prior_precision.to(self._device).to(self._dtype)

    def _sample_parameters(self):
        """Sample from the approximate posterior distribution."""
        # Start with the mean
        w_sample = self.mean.clone()
        
        # Add noise from diagonal component
        eps = torch.randn_like(w_sample)
        w_sample += eps / torch.sqrt(self.H)
        
        # Add low-rank component if available
        if self.U_full is not None and self.S_full is not None:
            rank = self.S_full.shape[0]
            z = torch.randn(rank, device=self._device)
            low_rank_sample = self.U_full @ (z * torch.sqrt(self.S_full))
            w_sample += low_rank_sample
            
        return w_sample

    def _set_params_with_vector(self, vec):
        offset = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = vec[offset:offset+param_size].view(param.shape)
            offset += param_size

    def sample(self, n_samples=100, generator=None):
        samples = torch.zeros(n_samples, self.mean.shape[0], device=self._device)
        
        for i in range(n_samples):
            samples[i] = self._sample_parameters()
            
        return samples

    def functional_variance(self, Js):
        """Compute predictive variance incorporating low-rank structure.
        
        Parameters
        ----------
        Js : torch.Tensor
            Jacobian of shape (batch, outputs, params)
            
        Returns
        -------
        variance : torch.Tensor
            Predictive variance of shape (batch, outputs, outputs)
        """
        # Start with diagonal variance component
        f_var = torch.zeros((Js.shape[0], Js.shape[1], Js.shape[1]), 
                        device=Js.device, dtype=Js.dtype)
        
        # Add diagonal component
        for i in range(Js.shape[0]):  # For each batch
            Ji = Js[i]  # (outputs, params)
            f_var[i] = Ji @ (1.0 / self.H).diag() @ Ji.T  # Ji H^-1 Ji^T
        
        # Add low-rank correction if available
        if self.U_full is not None and self.S_full is not None:
            for i in range(Js.shape[0]):
                Ji = Js[i]  # (outputs, params)
                JiU = Ji @ self.U_full  # (outputs, rank)
                low_rank = JiU @ (self.S_full.diag()) @ JiU.T  # (outputs, outputs)
                f_var[i] += low_rank
                
        return f_var

    def functional_covariance(self, Js):
        """Compute joint predictive covariance incorporating low-rank structure.
        
        Parameters
        ----------
        Js : torch.Tensor
            Jacobian of shape (batch*outputs, params)
            
        Returns
        -------
        covariance : torch.Tensor
            Joint predictive covariance
        """
        # Start with diagonal component
        f_cov = Js @ (1.0 / self.H).diag() @ Js.T
        
        # Add low-rank correction if available
        if self.U_full is not None and self.S_full is not None:
            JsU = Js @ self.U_full  # (batch*outputs, rank)
            low_rank = JsU @ (self.S_full.diag()) @ JsU.T
            f_cov += low_rank
                
        return f_cov
        
    def _nn_predictive_mean(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def _nn_predictive_variance(self, x, n_samples=100):
        self.model.eval()
        
        # Get the base prediction
        f_base = self._nn_predictive_mean(x)
        
        # Initialize storage for squared differences
        sum_sq_diff = torch.zeros_like(f_base)
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample parameters
                w_sample = self._sample_parameters()
                
                # Set model parameters
                self._set_params_with_vector(w_sample)
                
                # Get prediction
                f = self.model(x)
                
                # Accumulate squared difference
                sum_sq_diff += (f - f_base) ** 2
        
        # Reset model to mean parameters
        self._set_params_with_vector(self.mean)
        
        # Return variance estimate
        return sum_sq_diff / n_samples

    def _glm_predictive_distribution(self, X, joint=False, diagonal_output=True):
        Js, f_mu = self.backend.jacobians(X, enable_backprop=self.enable_backprop)
        
        if joint:
            f_mu = f_mu.flatten()  # (batch*out)
            f_var = self.functional_covariance(Js)  # (batch*out, batch*out)
        else:
            f_var = self.functional_variance(Js)  # (batch, out, out)
            if diagonal_output and f_var.ndim == 3:
                # Extract diagonal elements for each output dimension
                f_var = torch.diagonal(f_var, dim1=1, dim2=2)
        
        # Apply temperature scaling - this is critical for calibrated uncertainty
        f_mu = f_mu / self.temperature
        f_var = f_var / (self.temperature**2)
        
        return (f_mu.detach(), f_var.detach()) if not self.enable_backprop else (f_mu, f_var)

    def predict(self, x, pred_type='glm', link_approx='probit', n_samples=100):
        return super().__call__(
            x, 
            pred_type=pred_type,
            link_approx=link_approx,
            n_samples=n_samples
        )

    def evaluate(self, data_loader: DataLoader, batch_size: int = None) -> float:
        if batch_size is not None and batch_size < data_loader.batch_size:
            # Create new DataLoader with smaller batch size
            new_loader = DataLoader(
                dataset=data_loader.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=data_loader.num_workers
            )
            data_loader = new_loader
        
        # Rest of evaluation code remains the same
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100. * correct / total
