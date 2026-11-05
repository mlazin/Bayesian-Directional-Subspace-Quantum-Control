"""Expected-improvement acquisition optimization for SAASBO-DSO."""

import warnings

import jax.numpy as jnp
from jax import lax, value_and_grad
from jax.scipy.stats import norm
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import qmc

class EIAcquisition:
  def __init__(self,bounds,num_init_evals,seed=None,grad = None):

    self.bounds = bounds
    self.seed = seed
    self.grad = grad


  def ei(self,x, y_target, gp, xi=0.1):
    mu, var = gp.posterior(x)
    std = jnp.maximum(jnp.sqrt(var), 1e-6)
    improve = y_target - xi - mu
    scaled = improve / std
    cdf, pdf = norm.cdf(scaled), norm.pdf(scaled)
    exploit = improve * cdf
    explore = std * pdf
    values = jnp.nan_to_num(exploit + explore, nan=0.0)
    return  values.mean(axis=0)


  def ei_grad(self,x, y_target, gp, xi=0.1):
    # Gradient of EI
    return self.ei(x, y_target, gp, xi).sum()

  # ei_grad used only in negative_ei_and grad_with_penalty, in value_and_grad function as argument;
  # this negative.... function uses penalty and grad penalty, so that ei_grad should not have a seperate penalty
  def optimize_ei(self,gp, y_target, bounds, xi=0.1, num_restarts_ei=5, num_init=5000):
      # Helper function for optimizing EI


      def negative_ei_and_grad(x, y_target, gp, xi):
          # Compute EI and its gradient and then flip the signs since L-BFGS-B minimizes
          x = jnp.array(x.copy())[None, :]
          ei_val, ei_val_grad = value_and_grad(self.ei_grad)(x, y_target, gp, xi)                #can instead change here for gradient with penalty
          ret = ( -1 * ei_val.item(), -1 * np.array(ei_val_grad))
          return ret

      def negative_ei_and_grad_with_penalty(x, y_target, gp, xi, radius):
          x = jnp.array(x.copy())[None, :]
          ei_val, ei_val_grad = value_and_grad(self.ei_grad)(x, y_target, gp, xi)
          penalty = 10 * max(0, np.sum(x[0]**2) - radius**2)
          penalty_grad = 20 * x[0] if np.sum(x[0]**2) > radius**2 else np.zeros_like(x[0])
          return -ei_val.item() + penalty, -np.array(ei_val_grad).squeeze() + penalty_grad

      dim = gp.X_train.shape[1]

      tmpbounds = bounds
      #change sphere_bounds
      if isinstance(bounds,tuple):
        radius_bounds = bounds[1]
      else:
        radius_bounds = None
      if isinstance(bounds,tuple):
        if bounds[0].shape[1] > dim:   
          bounds = bounds[0][:,:dim]

      else:
        if bounds.shape[1] > dim:   
          bounds = bounds[:,:dim]
      # done sphere_bounds

      with warnings.catch_warnings(record=True):  # Suppress qmc.Sobol UserWarning

          #change sphere_bounds
          if isinstance(tmpbounds,tuple):
            ndim = tmpbounds[0].shape[-1]
            X_rand = qmc.Sobol(dim, scramble=True).random(num_init)
            X_rand /= np.sqrt((X_rand**2).sum(axis=1,keepdims=True))
            X_rand *= np.random.rand(num_init,1)**(1/ndim)*tmpbounds[1]
          else:
            X_rand = qmc.Sobol(dim, scramble=True).random(num_init)*(bounds[1]-bounds[0])+bounds[0]
          #done sphere_bounds


      # Make sure x_best is in the set of candidate EI maximizers
      if self.grad:
        x_best = gp.X_train[np.argmin(gp.Y_train[:,0]), :]   #added gradient
      else:
        x_best = gp.X_train[gp.Y_train.argmin(), :]

      print("from optimize ei:",x_best.shape,np.random.randn(1,dim).shape,(bounds[1]-bounds[0]).shape)

      if isinstance(tmpbounds,tuple):
        X_rand[0,:] = np.clip(x_best + 0.001*(bounds[0][1]-bounds[0][0])*np.random.randn(1,dim),a_min=bounds[0][0],a_max=bounds[0][1])
      else:
        X_rand[0,:] = np.clip(x_best + 0.001*(bounds[1]-bounds[0])*np.random.randn(1,dim),a_min=bounds[0],a_max=bounds[1])


      X_rand = jnp.array(X_rand)
      ei_rand = self.ei(X_rand, y_target, gp)
      _, top_inds = lax.top_k(ei_rand, num_restarts_ei)
      X_init = X_rand[top_inds, :]

      x_best, y_best = None, -float("inf")

      #change sphere_bounds for penalty

      if isinstance(tmpbounds,tuple):

          if isinstance(bounds,tuple):
            workingbounds = np.array(bounds[0])
          else:
            workingbounds = bounds
          for x0 in X_init:
              x, fx, d = fmin_l_bfgs_b(
                  func=negative_ei_and_grad_with_penalty,
                  x0=x0,
                  fprime=None,  # because the function returns (value, grad)
                  bounds=list(zip(workingbounds[0], workingbounds[1])),
                  args=(y_target, gp, 0.1, radius_bounds),  # <-- pass radius here
                  maxfun=5000
              )
              fx = -1 * fx  # Back to maximization
              if fx > y_best:
                  x_best, y_best = x.copy(), fx

      else:

            for x0 in X_init:

              x, fx, d = fmin_l_bfgs_b(
                  func=negative_ei_and_grad,
                  x0=x0,
                  fprime=None,
                  bounds = list(zip(bounds[0],bounds[1])),
                  args=(y_target, gp, 0.1),
                  maxfun=5000,  # this limits computational cost
                  #pgtol = 1e-10  # stricter tolerance for better optimization


              )
              fx = -1 * fx  # Back to maximization
              if fx > y_best:
                  x_best, y_best = x.copy(), fx

      return x_best, y_best

  def gp_posterior_mean_opt(self,xX, gp22):
      if len(xX.shape) == 1:
          xX = xX[None,:]
      m,v = gp22.posterior(xX)
      return np.mean(m)
