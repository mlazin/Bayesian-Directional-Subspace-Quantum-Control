"""SAAS Gaussian-process model used by SAASBO-DSO."""

import math
import time
from functools import partial

import jax
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as random
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

def get_chunks(L, chunk_size):
    num_chunks = L // chunk_size
    chunks = [jnp.arange(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    if L % chunk_size != 0:
        chunks.append(np.arange(L - L % chunk_size, L))
    return chunks


# fun should return tuples of arrays
def chunk_vmap(fun, array, chunk_size=4):
    L = array[0].shape[0]
    chunks = get_chunks(L, chunk_size)
    results = [vmap(fun)(*tuple([a[chunk] for a in array])) for chunk in chunks]
    num_results = len(results[0])
    return tuple([jnp.concatenate([r[k] for r in results]) for k in range(num_results)])



root_five = math.sqrt(5.0)
five_thirds = 5.0 / 3.0

# compute diagonal component of kernel
def kernel_diag(var, noise, jitter=1.0e-6, include_noise=True):
    if include_noise:
        return var + noise + jitter
    else:
        return var + jitter

#the fixed
@partial(jit, static_argnums=(5,))
def rbf_kernel_grad(X, Z, var, inv_length_sq, noise, include_noise):

    #Z = Z[:2,]
    N_X, P = X.shape
    N_Z = Z.shape[0]

    invlen = jnp.sqrt(inv_length_sq)

    # Scale inputs by length scale
    X_scaled = X * invlen
    Z_scaled = Z * invlen

    # Compute the squared distance
    deltaXsq = jnp.square(X_scaled[:, None, :] - Z_scaled[None,:,:])  # shape (N_X, N_Z, P)
    K_11 = var * jnp.exp(-0.5 * jnp.sum(deltaXsq, axis=-1))  # shape (N_X, N_Z)

    # Initialize full kernel matrix
    K = jnp.zeros((N_X * (P + 1), N_Z * (P + 1)))

    # Fill in the kernel block (function value covariance)
    K = K.at[:N_X, :N_Z].set(K_11)

    # Compute the gradient blocks
    outer1 = (X_scaled[:, None, :] - Z_scaled) * invlen # shape (N_X, N_Z, P)
    grad_block_1 = outer1 * K_11[..., None]  # shape (N_X, N_Z, P)

    # below converts to same ordering as gpy:
    grad_block_1 = grad_block_1.transpose(0,2,1)

    K = K.at[:N_X, N_Z:].set(grad_block_1.reshape(N_X, N_Z * P))

    # Transpose gradient block for symmetry
    outer2 = (Z_scaled[:,None,:] - X_scaled) * invlen # shape (N_Z,N_X,P)

    outer2 = outer2.transpose(1,0,2) # shape (N_X,N_Z,P)

    grad_block_2 = outer2 * K_11[...,None]
    grad_block_2 = grad_block_2.transpose(0,2,1)

    # below converts to same ordering as gpy:
    grad_block_2 = grad_block_2.transpose(1,0,2)

    K = K.at[N_X:, :N_Z].set(grad_block_2.reshape(N_X*P,N_Z))

    outer3 = -outer1[:,:,None,:]*outer2[:,:,:,None] # shape: [N_X,N_Z,P,P]


    identity_term = jnp.eye(P)*inv_length_sq # shape [P,P]

    hessian_block = (identity_term[None,None,:,:]-outer3)*K_11[:,:,None,None]
    hessian_block = jnp.permute_dims(hessian_block,(0,2,1,3))

    # below converts to same ordering as gpy:
    hessian_block = jnp.permute_dims(hessian_block,(1,0,3,2))

    K = K.at[N_X:, N_Z:].set(hessian_block.reshape(N_X*P,N_Z*P))

    """
    # Shuffle
    pi1 = np.arange(N_X * (P + 1)).reshape(P + 1, N_X).T.flatten()
    pi2 = np.arange(N_Z * (P + 1)).reshape(P + 1,  N_Z).T.flatten()
    K = K[..., pi1, :][..., :, pi2]

    """

    # Add noise

    if include_noise:
          K = K + (noise + 1e-6) * jnp.eye(N_X * (P + 1))

    return K


# X, Z have shape (N_X, P) and (N_Z, P)
@partial(jit, static_argnums=(5,))
def rbf_kernel(X, Z, var, inv_length_sq, noise, include_noise):
    deltaXsq = jnp.square(X[:, None, :] - Z) * inv_length_sq  # N_X N_Z P
    k = var * jnp.exp(-0.5 * jnp.sum(deltaXsq, axis=-1))
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z


# X, Z have shape (N_X, P) and (N_Z, P)
@partial(jit, static_argnums=(5,))
def matern_kernel(X, Z, var, inv_length_sq, noise, include_noise):
    deltaXsq = jnp.square(X[:, None, :] - Z) * inv_length_sq  # N_X N_Z P
    dsq = jnp.sum(deltaXsq, axis=-1)  # N_X N_Z
    exponent = root_five * jnp.sqrt(jnp.clip(dsq, a_min=1.0e-12))
    poly = 1.0 + exponent + five_thirds * dsq
    k = var * poly * jnp.exp(-exponent)
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z

key = jax.random.PRNGKey(0)
class SAASGP(object):
    """
    This class contains the necessary modeling and inference code to fit a gaussian process with a SAAS prior.

    See below for arguments.
    """

    def __init__(
        self,
        alpha=0.1,  # controls sparsity
        num_warmup=512,  # number of HMC warmup samples
        num_samples=256,  # number of post-warmup HMC samples
        max_tree_depth=7,  # max tree depth used in NUTS
        num_chains=1,  # number of MCMC chains
        thinning=16,  # thinning > 1 reduces the computational cost at the risk of less robust model inferences
        verbose=True,  # whether to use stdout for verbose logging
        observation_variance=0.0,  # observation variance to use; this scalar value is inferred if observation_variance==0.0
        kernel="rbf_grad",  # GP kernel to use (matern or rbf)
        baselenscale = 1.0, # CHANGE2
    ):
        if alpha <= 0.0:
            raise ValueError("The hyperparameter alpha should be positive.")
        if observation_variance < 0.0:
            raise ValueError("The hyperparameter observation_variance should be non-negative.")
        if kernel not in ["matern", "rbf", "rbf_grad"]:
            raise ValueError("Allowed kernels are matern and rbf.")
        for i in [num_warmup, num_samples, max_tree_depth, num_chains, thinning]:
            if not isinstance(i, int) or i <= 0:
                raise ValueError(
                    "The hyperparameters num_warmup, num_samples, max_tree_depth, "
                    + "num_chains, and thinning should be positive integers."
                )

        self.alpha = alpha
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.thinning = thinning
        self.verbose = verbose
        self.observation_variance = observation_variance
        self.learn_noise = observation_variance == 0.0
        self.Ls = None
        self.count = 1
        self.baselenscale = baselenscale # CHANGE2

        if kernel == "rbf":
            self.kernel = rbf_kernel
        elif kernel == "rbf_grad":
            self.kernel = rbf_kernel_grad
        else:
            self.kernel = matern_kernel



    ####changes for type
    def ensure_array(self,k):
      return jax.device_get(k) if isinstance(k, jax.core.Tracer) else k

    ###
    ### changes for valid cov
    def fix_covariance_matrix(self,k):
        # Ensure symmetry
        k = (k + k.T) / 2

        # Get eigenvalues to check positive semi-definiteness
        eigenvalues = jnp.linalg.eigvalsh(k)

        # Add jitter until the matrix is positive semi-definite
        min_eigenvalue = jnp.min(eigenvalues)
        jitter_needed = jnp.maximum(0.0, -min_eigenvalue + 1e-6)
        k += jnp.eye(k.shape[0]) * jitter_needed

        return k

    ###
    # define the surrogate model. users who want to modify e.g. the prior on the kernel variance
    # should make their modifications here.


    def model(self, X, Y):

        self.count = self.count + 1

        N, P = X.shape

        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0),rng_key = key)
        #var = jax.device_get(var)
        #var = self.ensure_array(var)
        noise = (
            numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0)) if self.learn_noise else self.observation_variance
        )
        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(self.alpha))

        # note we use deterministic to reparameterize the geometry
        inv_length_sq = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(P)))
        # CHANGE2
        inv_length_sq = numpyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq / (self.baselenscale**2))

        k = self.kernel(X, X, var, inv_length_sq, noise, True)


        ress = dist.MultivariateNormal(loc=jnp.zeros(k.shape[0]), covariance_matrix=k)
        #numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.zeros(k.shape[0]), covariance_matrix=k), obs=Y.T.reshape((-1,)))
        numpyro.sample("Y", ress, obs=Y.T.reshape((-1,)))
        
        
    # run gradient-based NUTS MCMC inference
    def run_inference(self, rng_key, X, Y):
        self.count = self.count + 1
        start = time.time()
        kernel = NUTS(self.model, max_tree_depth=self.max_tree_depth)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=self.verbose,
        )

        mcmc.run(rng_key, X, Y)
        flat_samples = mcmc.get_samples(group_by_chain=False)
        chain_samples = mcmc.get_samples(group_by_chain=True)
        flat_summary = summary(flat_samples, prob=0.90, group_by_chain=False)

        if self.verbose:
            rhat = flat_summary["kernel_inv_length_sq"]["r_hat"]
            print(
                "[kernel_inv_length_sq] r_hat min/max/median:  {:.3f}  {:.3f}  {:.3f}".format(
                    np.min(rhat), np.max(rhat), np.median(rhat)
                )
            )

            mcmc.print_summary(exclude_deterministic=False)
            print("\nMCMC elapsed time:", time.time() - start)

        return chain_samples, flat_samples, flat_summary
    

    # compute cholesky factorization of kernel matrices (necessary to compute posterior predictions)
    def compute_choleskys(self, chunk_size=8):
        def _cholesky(var, inv_length_sq, noise):
            k_XX = self.kernel(self.X_train, self.X_train, var, inv_length_sq, noise, True)
            return (cho_factor(k_XX, lower=True)[0],)

        n_samples = (self.num_samples * self.num_chains) // self.thinning
        vmap_args = (
            self.flat_samples["kernel_var"][:: self.thinning],
            self.flat_samples["kernel_inv_length_sq"][:: self.thinning],
            self.flat_samples["kernel_noise"][:: self.thinning]
            if self.learn_noise
            else self.observation_variance * jnp.ones(n_samples),
        )

        self.Ls = chunk_vmap(_cholesky, vmap_args, chunk_size=chunk_size)[0]

    # make predictions at test points X_test for a single set of SAAS hyperparameters
    def predict(self, rng_key, X, Y, X_test, L, var, inv_length_sq, noise, jointvar=False):

        testn = X_test.shape[0]

        k_pX = self.kernel(X_test, X, var, inv_length_sq, noise, False)
        #testcho = cho_solve((L[:Y.shape[1],:Y.shape[1]], True),Y.T)
        testcho = cho_solve((L,True),Y.T.flatten())
        #mean = jnp.matmul(k_pX, cho_solve((L, True), Y))
        mean = jnp.matmul(k_pX, testcho)
        if not jointvar:

            k_pp = kernel_diag(var, noise, include_noise=True)

            L_kXp = solve_triangular(L, jnp.transpose(k_pX), lower=True)

            diag_cov = k_pp - (L_kXp * L_kXp).sum(axis=0)

            return mean[:testn], diag_cov[:testn]
        else:
            k_pp = self.kernel(X_test,X_test,var,inv_length_sq,noise,True)
            L_kXp = solve_triangular(L,jnp.transpose(k_pX), lower=True)
            cov = k_pp - jnp.matmul(jnp.transpose(L_kXp),L_kXp)
            return mean[:testn], cov[:testn,:testn]

    def return_inv_length_scale(self):
      return self.flat_samples["kernel_inv_length_sq"][:: self.thinning]

    # fit SAASGP to training data
    def fit(self, X_train, Y_train, seed=0):
        self.X_train, self.Y_train = X_train.copy(), Y_train.copy()
        self.rng_key_hmc, self.rng_key_predict = random.split(random.PRNGKey(seed), 2)
        print("from saasgp line 350:", X_train.shape, Y_train.shape)
        self.chain_samples, self.flat_samples, self.summary = self.run_inference(self.rng_key_hmc, X_train, Y_train)

        return self

    # compute predictions at X_test using inferred SAAS hyperparameters
    def posterior(self, X_test, jointvar=False):

        if self.Ls is None:
            self.compute_choleskys(chunk_size=8)

        n_samples = (self.num_samples * self.num_chains) // self.thinning
        vmap_args = (
            random.split(self.rng_key_predict, n_samples),
            self.flat_samples["kernel_var"][:: self.thinning],
            self.flat_samples["kernel_inv_length_sq"][:: self.thinning],
            self.flat_samples["kernel_noise"][:: self.thinning] if self.learn_noise else 1e-6 * jnp.ones(n_samples),
            self.Ls,
        )

        predict = lambda rng_key, var, inv_length_sq, noise, L: self.predict(
            rng_key, self.X_train, self.Y_train, X_test, L, var, inv_length_sq, noise, jointvar
        )

        mean, var = chunk_vmap(predict, vmap_args, chunk_size=8)

        return mean, var
