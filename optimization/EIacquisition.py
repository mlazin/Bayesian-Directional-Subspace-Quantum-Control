"""Expected-improvement acquisition functions used by BO-DSO."""

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

def _make_x_for_model(x_np, model):
    p = next(model.parameters())
    return torch.tensor(x_np, dtype=p.dtype, device=p.device, requires_grad=True)

def model_negmean(x, model, likelihood, best_y=None, grad=True, xi=0.1):
    """
    Negative predictive mean.
    Same interface as before, but now supports gradients so SciPy
    """
    if not grad:
        x_t = torch.from_numpy(x)
        model.eval()
        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)
        with torch.enable_grad():
            pred = model(x_t)
        mu = pred.mean.detach().numpy()
        return -mu[0, 0]
    x_t = _make_x_for_model(x, model)
    model.eval()
    if likelihood is not None:
        likelihood.eval()
    model.zero_grad(set_to_none=True)
    if likelihood is not None:
        likelihood.zero_grad(set_to_none=True)
    with torch.enable_grad(), gpytorch.settings.fast_pred_var():
        pred = model(x_t)
        mu_all = pred.mean
        if mu_all.ndim == 1:
            mu_all = mu_all.view(1, -1)
        mu = mu_all[:, 0].squeeze()
        obj = -mu
        obj.backward()
    if x_t.grad is None:
        raise RuntimeError('model_negmean: x_t.grad is None (gradient did not flow)')
    grad_np = x_t.grad.detach().cpu().numpy().reshape(-1)
    val = float(obj.detach().cpu().numpy())
    return (val, grad_np)
_SQRT_2 = float(np.sqrt(2.0))
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))

def _normal_pdf_torch(z):
    return torch.exp(-0.5 * z ** 2) / _SQRT_2PI

def _normal_cdf_torch(z):
    return 0.5 * (1.0 + torch.erf(z / _SQRT_2))

def expected_improvement(x, model, likelihood, best_y, grad=True, xi=0.1):
    """
    - If grad=True: x comes in as numpy (shape (d,)) from SciPy.
      We create a torch tensor with requires_grad=True and return (value, grad_numpy).
    - If grad=False: keep old behavior (returns numpy array) for plotting code.
    """
    if not grad:
        x_t = torch.from_numpy(x)
        model.eval()
        likelihood.eval()
        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)
        pred = model(x_t)
        mu = pred.mean.detach().numpy()
        std = pred.variance.sqrt().detach().numpy()
        imp = best_y - mu - xi
        Z = imp / np.maximum(std, 1e-12)
        from scipy.stats import norm
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        return ei
    x_t = _make_x_for_model(x, model)
    model.eval()
    if likelihood is not None:
        likelihood.eval()
    model.zero_grad(set_to_none=True)
    if likelihood is not None:
        likelihood.zero_grad(set_to_none=True)
    with torch.enable_grad(), gpytorch.settings.fast_pred_var():
        if x_t.shape[0] != 1:
            x_t_input = x_t.view(1, -1)
        else:
            x_t_input = x_t
        pred = model(x_t_input)
        mu_all = pred.mean
        var_all = pred.variance
        if mu_all.ndim == 1:
            mu_all = mu_all.view(1, -1)
        if var_all.ndim == 1:
            var_all = var_all.view(1, -1)
        mu = mu_all[:, 0].squeeze()
        var = var_all[:, 0].squeeze().clamp_min(1e-12)
        std = torch.sqrt(var)
        imp = best_y - xi - mu
        Z = imp / std
        ei = imp * _normal_cdf_torch(Z) + std * _normal_pdf_torch(Z)
        ei = torch.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
        obj = -ei
        obj.backward()
    if x_t.grad is None:
        raise RuntimeError('EI: x_t.grad is None (gradient did not flow)')
    grad_np = x_t.grad.detach().cpu().numpy().reshape(-1)
    val = float(obj.detach().cpu().numpy())
    return (val, grad_np)

def optimize_acquisition(acquisition_func, model, likelihood, target, func_bounds, grad=True, n=2, xvals=None, yvals=None, plot=True, retval=False):
    bounds = func_bounds
    bounds = [(bounds[0][i], bounds[1][i]) for i in range(n)]
    b = np.array(bounds)
    if n == 1 and plot:
        print('plotting')
        fig, axs = plt.subplots(2, 1)
        xs = np.linspace(bounds[0][0], bounds[0][1], 100)
        model.eval()
        likelihood.eval()
        pred = model(torch.from_numpy(xs))
        mxs = pred.mean.detach().numpy()
        stds = pred.variance.sqrt().detach().numpy()
        if grad:
            mxs = mxs[:, 0]
            stds = stds[:, 0]
        imp = target - mxs - 0.1
        Z = imp / stds
        ei = imp * norm.cdf(Z) + stds * norm.pdf(Z)
        axs[0].plot(xs, ei, 'k-')
        axs[1].plot(xs, mxs, 'b-')
        axs[1].plot(xs, mxs + stds, 'b--')
        axs[1].plot(xs, mxs - stds, 'b--')
        if xvals is not None and yvals is not None:
            eps = 0.05
            axs[1].plot(xvals, yvals[:, 0], 'ko')
            for i in range(len(xvals)):
                axs[1].plot([xvals[i] - eps, xvals[i] + eps], [yvals[i, 0] - yvals[i, 1] * eps, yvals[i, 0] + yvals[i, 1] * eps], 'k--')
        axs[0].set_xlim(bounds[0])
        axs[1].set_xlim(bounds[0])
        plt.show()
    elif n == 2 and plot:
        axs = [0, 0]
        x1s = np.linspace(-0.5, 0.0, 100)
        x2s = np.linspace(-0.5, 0.0, 100)
        X1, X2 = np.meshgrid(x1s, x2s)
        X = np.hstack((X1.flatten()[:, None], X2.flatten()[:, None]))
        model.eval()
        likelihood.eval()
        pred = model(torch.from_numpy(X))
        mxs = pred.mean.detach().numpy()
        stds = pred.variance.sqrt().detach().numpy()
        if grad:
            mxs = mxs[:, 0]
            stds = stds[:, 0]
        imp = target - mxs - 0.1
        Z = imp / stds
        ei = imp * norm.cdf(Z) + stds * norm.pdf(Z)
        if False:
            fig = plt.figure()
            axs = [0, 0]
            axs[0] = fig.add_subplot(2, 1, 1, projection='3d')
            axs[1] = fig.add_subplot(2, 1, 2, projection='3d')
            axs[0].plot_surface(X1, X2, ei.reshape(X1.shape))
            axs[1].plot_surface(X1, X2, mxs.reshape(X1.shape), alpha=0.5)
            if xvals is not None and yvals is not None:
                axs[1].plot(xvals[:, 0], xvals[:, 1], yvals[:, 0], 'ko')
        else:
            fig, axs = plt.subplots(2, 1)
            axs[0].contour(X1, X2, ei.reshape(X1.shape), levels=10)
            axs[1].contour(X1, X2, mxs.reshape(X1.shape), levels=10)
            if xvals is not None and yvals is not None:
                axs[1].plot(xvals[:, 0], xvals[:, 1], 'ko')
        axs[0].set_xlim([-0.5, 0.0])
        axs[0].set_ylim([-0.5, 0.0])
        axs[1].set_xlim([-0.5, 0.0])
        axs[1].set_ylim([-0.5, 0.0])
        plt.show()
    ret = None
    rety = None
    if xvals is not None:
        besti = np.argmin(yvals if yvals.ndim == 1 else yvals[:, 0])
        bestx = xvals[besti]
    else:
        bestx = None
    nrestart = 10 * n if n < 10 else 20
    print('nrestart:', nrestart)
    cbi = 0

    def cb(intres):
        nonlocal cbi
        print(f'subit # {cbi}')
        cbi += 1
    for i in range(nrestart):
        print(f'starting iteration {i}')
        if i == 0 and bestx is not None:
            x0 = np.random.rand(n) * (b[:, 1] - b[:, 0]) + b[:, 0]
        else:
            x0 = np.random.rand(n) * (b[:, 1] - b[:, 0]) + b[:, 0]

        def fun(x):
            val, _ = acquisition_func(x, model, likelihood, target, grad=True, xi=0.1)
            return val

        def jac(x):
            _, g = acquisition_func(x, model, likelihood, target, grad=True, xi=0.1)
            return g
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', bounds=bounds, callback=cb)
        if ret is None or rety > res.fun:
            ret = res.x
            rety = res.fun
    if retval:
        return (ret, -rety)
    else:
        return ret
