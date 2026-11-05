"""Gaussian-process models used by BO-DSO."""

import gpytorch


class MyGPGD(gpytorch.models.ExactGP):
    """Gradient-augmented GP with an ARD Matérn-5/2 kernel."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.Matern52KernelGrad(
                ard_num_dims=train_x.shape[1]
            )
        )
        self.covar_module.base_kernel.lengthscale = 0.1

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)


class MyGPNOGD(gpytorch.models.ExactGP):
    """Standard GP without derivative observations."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.Matern52Kernel(
                ard_num_dims=train_x.shape[1]
            )
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
