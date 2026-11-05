"""SAASBO optimizers used by the quantum-control experiments."""

#saasopt sphere_bounds #saasbomodigDir
import numpy as np
from .saasgp import SAASGP, rbf_kernel_grad
from .saasboecq import EIAcquisition
import matplotlib.pyplot as plt

import time
from copy import deepcopy
import numpyro
from numpyro import enable_x64

def _flatten_grad(g):
    g = np.asarray(g)
    return g.reshape(-1)

class saasbomodigDir:
    def __init__(self, bounds, num_init_evals, seed=None, maxiter=10,
                 grad=True, gradcomb=True, alpha=0.1, num_warmup=512, num_samples=256,
                 thinning=16, num_restarts_ei=5, device="cpu"):
        self.bounds = bounds
        # Precompute shape info from ctor bounds (may be overridden per-run by f.bounds())
        if isinstance(self.bounds, tuple):
            self.lb  = self.bounds[0].shape[1]
            self.lbs = self.bounds[0][0]
            self.ubs = self.bounds[0][1]
        else:
            self.lb  = self.bounds.shape[1]
            self.lbs = self.bounds[0]
            self.ubs = self.bounds[1]

        self.num_init_evals = num_init_evals
        self.seed = seed
        self.maxiter = maxiter
        self.alpha = alpha
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.thinning = thinning
        self.num_restarts_ei = num_restarts_ei
        self.grad = grad
        self.gradcomb = gradcomb
        self.device = device

        # state
        self.count = 1
        self.x = 0
        self.y = 0
        self.bestx = 0           # best input in *alpha*-space (training min)
        self.bestX_full = None   # best X in full space when f has X0/D
        self.x_next = 0
        self.bestysofar = 0
        self.loop = 1
        self.fixedcount = 0
        self.bestP = 0
        self.xnexts = []

        self.kernel = "rbf_grad" if self.grad else "rbf"

    def plot(self, gp, optei, Xpts, Yvals, ytarget, Ygrads=None):
        import matplotlib.pyplot as plt
        print('starting plot')
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        npts = 15
        X = np.linspace(0, 1e-2, npts)
        Y = np.linspace(0, 1e-2, npts)
        X, Y = np.meshgrid(X, Y)
        Z1 = np.zeros_like(X)
        Z2 = np.zeros_like(X)
        for i, (x, y) in enumerate(zip(X, Y)):
            for j, (xx, yy) in enumerate(zip(x, y)):
                pt = np.array([[xx, yy]])
                mu, var = gp.posterior(pt)
                Z1[i, j] = mu.mean()
                eival = optei.ei(pt, ytarget, gp)
                Z2[i, j] = eival[0]

        Z1 = Z1 * Yvals.std() + Yvals.mean()
        ax1.plot_surface(X, Y, Z1)
        ax1.contour(X, Y, Z1, zdir='z', offset=np.min(Z1))
        ax1.set_title('GP posterior mean')
        ax1.plot(Xpts[:, 0], Xpts[:, 1], Yvals, 'ko')
        ax1.view_init(elev=30, azim=-60)

        ax2.plot_surface(X, Y, Z2)
        ax2.contour(X, Y, Z2, zdir='z', offset=np.min(Z2))
        ax2.set_title('acquisition function')
        ax2.plot(Xpts[:, 0], Xpts[:, 1], np.zeros_like(Xpts[:, 0]), 'ko')
        ax2.view_init(elev=30, azim=-60)

        with open('midopt.npy', 'wb') as f:
            np.save(f, X)
            np.save(f, Y)
            np.save(f, Z1)
            np.save(f, Z2)

        plt.show()

    def _construct_full_X_if_possible(self, f, a):
        if not hasattr(f, 'X0') or not hasattr(f, 'D'):
            return None
        X0 = np.asarray(f.X0, float).reshape(-1)
        D  = np.asarray(f.D, float)
        a  = np.asarray(a, float).reshape(-1)

        if D.ndim == 1:  # 1-D direction
            return X0 + D * float(a[0])
        # D has shape (D_full, r). Match a's length if possible.
        if a.size == D.shape[1]:
            return X0 + D.dot(a)
        # If a is length-1 but D has multiple columns, use the first column by convention
        return X0 + D[:, 0] * float(a[0])
        
        
    ########## invalid kernel matrix error check ###############
    def rbf_is_symmetric_and_psd(self,X, lengthscale=1.0, variance=1.0, atol=1e-10):
      """
      Build the RBF kernel on X
      symmetry: K == K.T within atol
      PSD: all eigenvalues >= -atol (tolerant for tiny numerical negatives)
      Returns True only if both hold.
      """
      K = rbf_kernel_grad(X,X,variance,lengthscale,noise=1e-12,include_noise=True)#rbf_kernel(X, lengthscale, variance)

      #symmetry check
      is_sym = np.allclose(K, K.T, rtol=0.0, atol=atol)

      #PSD check via eigenvalues of symmetric matrix

      Ks = 0.5 * (K + K.T)
      eigs = np.linalg.eigvalsh(Ks)
      is_psd = eigs.min() >= -atol

      return bool(is_sym and is_psd)


    def check_inputs(self,X, Y):
        X = np.asarray(X); Y = np.asarray(Y)
        if not np.all(np.isfinite(X)): raise ValueError("X has NaN/Inf.")
        if not np.all(np.isfinite(Y)): raise ValueError("Y has NaN/Inf.")
        # exact duplicates in X ? near-singular K
        if len(np.unique(X, axis=0)) != len(X):
            print("Warning: duplicate rows in X found; consider deduping or adding noise.")
        
        return X, Y
    ########## end invalid kernel matrix error check ###############

    def run(self, f, initx, inity=None, initgrad=None, maxitt=1):
        self.loop = 1
        self.fixedcount = 0
        print("saasbo run started............")
        if self.alpha <= 0.0:
            raise ValueError("The hyperparameter alpha must be positive.")
        if self.device not in ["cpu", "gpu"]:
            raise ValueError("The device must be cpu or gpu.")

        # Resolve bounds for THIS RUN from f.bounds() if available (enables 1-D problems)
        if hasattr(f, 'bounds') and callable(f.bounds):
            bounds_f = np.asarray(f.bounds(), float)
        else:
            bounds_f = self.bounds

        # local shape helpers for bounds_f
        if isinstance(bounds_f, tuple):
            lb_local  = bounds_f[0].shape[1]
            lbs_local = bounds_f[0][0]
            ubs_local = bounds_f[0][1]
        else:
            lb_local  = bounds_f.shape[1]
            lbs_local = bounds_f[0]
            ubs_local = bounds_f[1]


        num_exceptions = 0

        # normalize input seed shapes 
        self.x = np.asarray(initx, float)
        if self.x.ndim == 1:
            self.x = self.x.reshape(1, -1)          # (N, k)

        if inity is not None:
            inity = np.asarray(inity, float).reshape(-1)  # (N,)

        if initgrad is not None:
            initgrad = np.asarray(initgrad, float)
            if initgrad.ndim == 1:
                initgrad = initgrad.reshape(1, -1)  # (N, k)

        ys, gs = [], []
        yvals, gvals = [], []

        print("initx shape:", getattr(self.x, 'shape', None), "inity shape:", (None if inity is None else inity.shape))
        if self.gradcomb:
            if inity is not None:
                # seed from provided init y/grad
                for i in range(self.x.shape[0]):
                    yi = float(inity[i])
                    gi = np.asarray(initgrad[i], float).reshape(-1)
                    ys.append(yi); yvals.append(yi)
                    gs.append(gi); gvals.append(gi)
            else:
                # evaluate seeds
                for i in range(self.x.shape[0]):
                    fval, g = f(self.x[i], grad=True)
                    yi = float(fval)
                    gi = np.asarray(g, float).reshape(-1)
                    ys.append(yi); yvals.append(yi)
                    gs.append(gi); gvals.append(gi)

            self.y = np.array(ys, float)                 # (N,)
            gs = np.array(gs, float)                     # (N, k)
            self.y = np.hstack((self.y.reshape(-1, 1), gs))  # (N, 1+k)
        else:
            if inity is None or len(np.asarray(inity).reshape(-1)) == 0:
                ys = []
                for i in range(self.x.shape[0]):
                    fval = f(self.x[i], grad=False)
                    ys.append(float(fval))
                self.y = np.array(ys, float)            # (N,)
            else:
                self.y = np.asarray(inity, float).reshape(-1)

        print("Starting SAASBO optimization run.")
        while self.loop > 0:
        #for h in range(self.maxiter):                 #change for fixed iter
            print(f"=== Iteration {len(self.y)} ===", flush=True)

            # standardize training data
            if self.gradcomb:
                ymean = self.y[:, 0].mean()
                ystd = self.y[:, 0].std()
                train_Y = self.y.copy()
                train_Y[:, 0] -= ymean
                train_Y /= (ystd if ystd != 0 else 1.0)
                y_target = float(train_Y[:, 0].min())
            else:
                train_Y = (self.y - self.y.mean()) / (self.y.std() if self.y.std() != 0 else 1.0)
                y_target = float(train_Y.min())

            try:
                start = time.time()
                gp = SAASGP(
                    alpha=self.alpha,
                    num_warmup=self.num_warmup,
                    num_samples=self.num_samples,
                    max_tree_depth=6,
                    num_chains=1,
                    thinning=self.thinning,
                    verbose=False,
                    observation_variance=1e-6,
                    kernel=self.kernel,
                    baselenscale=(bounds_f[1] - bounds_f[0]).max()
                )

                
                # fit SAAS GP to training data
                print("start gp fitting........")
                gp = gp.fit(self.x, train_Y)
                idx = np.argmin(self.y[:, 0]) if self.gradcomb else np.argmin(self.y)
                self.bestx = gp.X_train[idx]  # best in alpha-space (training min)
                print(f"GP fitting took {time.time() - start:.2f} seconds")
                
                start = time.time()
                optei = EIAcquisition(bounds_f, self.num_init_evals, self.seed, grad=self.gradcomb)
                self.x_next, y_bnext = optei.optimize_ei(
                    gp=gp,
                    y_target=min(-0.9, y_target),
                    bounds=bounds_f,
                    xi=0.0,
                    num_restarts_ei=self.num_restarts_ei,
                    num_init=5000
                )
                print("x_next:", self.x_next.shape if hasattr(self.x_next, 'shape') else type(self.x_next), self.x_next)
                print(f"Optimizing EI took {time.time() - start:.2f} seconds")
                self.xnexts.append(self.x_next)

            except Exception as e:
                num_exceptions += 1
                print(f"WARNING: Exception was raised ({e}), using a random point.")
                self.x_next = np.random.rand(lb_local) * (ubs_local - lbs_local) + lbs_local
                if num_exceptions <= 10:
                    raise
                else:
                    raise RuntimeError("ERROR: Maximum number of exceptions raised!")

            del gp  # Free memory

            start = time.time()
            self.gradcomb = True
            if self.gradcomb:
                fval, g = f(self.x_next, grad=True)
                yvals.append(float(fval))
                gvals.append(np.asarray(g, float).reshape(-1))
                self.y = np.array(yvals, float)               # (M,)
                gs = np.array(gvals, float)                   # (M, k)
                self.y = np.hstack((self.y.reshape(-1, 1), gs))
                self.x = np.vstack((self.x, deepcopy(np.asarray(self.x_next)[None, :])))

            else:
                fval = f(self.x_next, grad=False)
                self.y = np.hstack((self.y, float(fval)))
                self.x = np.vstack((self.x, deepcopy(np.asarray(self.x_next)[None, :])))

            # best so far
            if self.gradcomb:
                idx = np.argmin(self.y[:, 0])
                self.bestysofar = self.y[idx, 0]
            else:
                idx = np.argmin(self.y)
                self.bestysofar = self.y[idx]

            diff = np.abs(fval) - np.abs(self.bestP)
            if diff > 1e-4:
                self.bestP = np.abs(fval)
            else:
                self.fixedcount += 1

            print("fixedcount, bestP , maxiter ", self.fixedcount, self.bestP, self.maxiter)
            if self.fixedcount == self.maxiter:
                self.loop = 0
            print("bestysofar:", self.bestysofar, self.loop)

        # Return bestX in full space if f has X0/D, else x_next
        if self.gradcomb and isinstance(self.x, np.ndarray) and self.x.shape[0] >= 1:
            idx = int(np.argmin(self.y[:, 0]))
            best_alpha = self.x[idx]
        else:
            idx = int(np.argmin(self.y)) if isinstance(self.y, np.ndarray) else -1
            best_alpha = self.x[idx] if idx >= 0 else self.x_next

        X_full = self._construct_full_X_if_possible(f, best_alpha)
        self.bestX_full = X_full

        return X_full if X_full is not None else self.x_next
        
        
class saasbomodi:
  def __init__(self,bounds,num_init_evals,seed=None,maxiter=10,grad=True,gradcomb = True,alpha=0.1,num_warmup=512,num_samples=256,
                thinning=16,num_restarts_ei=5,device="cpu"):

    self.bounds = bounds
    #change sphere_bounds
    if isinstance(self.bounds,tuple):
      self.lb = self.bounds[0].shape[1]
      self.lbs = self.bounds[0][0]
      self.ubs = self.bounds[0][1]
    else:
      self.lb = self.bounds.shape[1]
      self.lbs = self.bounds[0]
      self.ubs = self.bounds[1]
    self.num_init_evals = num_init_evals
    self.seed = seed
    self.maxiter = maxiter
    self.alpha = alpha
    self.num_warmup = num_warmup
    self.num_samples = num_samples
    self.thinning = thinning
    self.num_restarts_ei = num_restarts_ei
    #self.kernel = kernel
    self.grad = grad
    self.gradcomb = gradcomb
    self.device = device
    self.count = 1
    self.x = 0
    self.y = 0
    self.bestx = 0
    self.x_next = 0
    self.bestysofar = 0
    
    self.loop = 1
    self.fixedcount = 0
    self.bestP = 0
    
    self.xnexts = []

    if self.grad:
      self.kernel = "rbf_grad"
    else:
      self.kernel = "rbf"

  def plot(self,gp,optei,Xpts,Yvals,ytarget,Ygrads=None):
      print('starting plot')
      fig1 = plt.figure()
      ax1 = fig1.add_subplot(111,projection='3d')
      fig2 = plt.figure()
      ax2 = fig2.add_subplot(111,projection='3d')
      npts = 15
      X = np.linspace(0,1e-2,npts)
      Y = np.linspace(0,1e-2,npts)
      X,Y = np.meshgrid(X,Y)
      Z1 = np.zeros_like(X)
      Z2 = np.zeros_like(X)
      for i,(x,y) in enumerate(zip(X,Y)):
          for j,(xx,yy) in enumerate(zip(x,y)):
              pt = np.array([[xx,yy]])
              mu, var = gp.posterior(pt)
              Z1[i,j] = mu.mean()
              eival = optei.ei(pt,ytarget,gp)
              Z2[i,j] = eival[0]


      Z1 = Z1*Yvals.std() + Yvals.mean()
      ax1.plot_surface(X,Y,Z1)
      ax1.contour(X, Y, Z1, zdir='z', offset=np.min(Z1))
      ax1.set_title('GP posterior mean')
      ax1.plot(Xpts[:,0],Xpts[:,1],Yvals,'ko')
      #if Ygrads is not None:
      #    for xx,y,grad in zip(Xpts,Yvals,Ygrads):
      #        ax1.plot([xx[0],xx[0]+grad[0]],[xx[1],xx[1]+grad[1]],[y,y],'b-')
      ax1.view_init(elev=30, azim=-60)

      ax2.plot_surface(X,Y,Z2)
      ax2.contour(X, Y, Z2, zdir='z', offset=np.min(Z2))
      ax2.set_title('acquisition function')
      ax2.plot(Xpts[:,0],Xpts[:,1],np.zeros_like(Xpts[:,0]),'ko')
      ax2.view_init(elev=30, azim=-60)

      with open('midopt.npy','wb') as f:
          np.save(f,X)
          np.save(f,Y)
          np.save(f,Z1)
          np.save(f,Z2)

      plt.show()

  def run(self,f,initx,inity = None,initgrad = None, maxitt=1):

      self.loop = 1
      self.fixedcount = 0
      print("saasbo run started............")
      if self.alpha <= 0.0:
          raise ValueError("The hyperparameter alpha must be positive.")
      if self.device not in ["cpu", "gpu"]:
          raise ValueError("The device must be cpu or gpu.")

      def checkgrad(fval,g,x):
            # quick check of gradient:
            # good only for 2d (hardcoded)
            #####
            delx = 1e-6
            fval1 = f(x+np.array([1,0])*delx,grad=False)
            fval2 = f(x+np.array([0,1])*delx,grad=False)
            #print('returned grad:',g)
            #print('finite-diff g:',(np.array([fval1,fval2])-fval)/delx)

      numpyro.set_platform(self.device)
      enable_x64()
      numpyro.set_host_device_count(1)
      num_exceptions = 0

      self.x = initx
      if inity is not None:
        self.y = inity
      else:
        self.y = np.array([])

      ys = []
      gs = []
      yvals = []
      gvals = []
      if self.gradcomb:
        if inity is not None:          # value and gradient from expandopt
          for i in range(len(initx)):
              ys.append(inity[i])
              yvals.append(inity[i])   #multiple as the y,g ordering
              gvals.append(initgrad[i])
              gs.append(initgrad[i])
        else:
          for i in range(len(initx)):
              fval,g = f(initx[i],grad=True)
              #checkgrad(fval,g,initx[i])
              ys.append(fval)
              yvals.append(fval)
              gvals.append(g)
              gs.append(g)
        self.y = np.array(ys)
        gs = np.array(gs)
        self.y = np.hstack((self.y.reshape(-1,1),gs))
      else:
        if len(self.y) == 0:
          ys = []
          for i in range(len(initx)):
            fval = f(initx[i],grad=False)
            ys.append(fval)
          self.y = np.array(ys)

      print("Starting SAASBO optimization run.")
      # print(f"First {self.num_init_evals} queries drawn at random. Best minimum thus far: {self.y.min().item():.3f}")

      while self.loop > 0:
          print(f"=== Iteration {len(self.y)} ===", flush=True)
          # standardize training data
          if self.gradcomb:
            #print("gradcomb yes ....",self.maxiter)
            ymean = self.y[:,0].mean()
            ystd = self.y[:,0].std()
            train_Y = self.y.copy()
            train_Y[:,0] -= ymean
            train_Y /= ystd            
            y_target = train_Y[:,0].min().item()
          else:
             train_Y = (self.y - self.y.mean()) / self.y.std()             
             y_target = train_Y.min().item()

          # If for whatever reason we fail to return a query point above we choose one at random from the domain
          try:
              start = time.time()
              # define GP with SAAS prior
              gp = SAASGP(
                  alpha=self.alpha,
                  num_warmup=self.num_warmup,
                  num_samples=self.num_samples,
                  max_tree_depth=6,
                  num_chains=1,
                  thinning=self.thinning,
                  verbose=False,
                  observation_variance=1e-6,
                  kernel=self.kernel,
                  baselenscale = (self.bounds[1]-self.bounds[0]).max() #CHANGE2
              )

              # fit SAAS GP to training data
              gp = gp.fit(self.x, train_Y)
              if self.gradcomb:
                idx = np.argmin(self.y[:,0])
              else:
                idx = np.argmin(self.y)
              self.bestx = gp.X_train[idx]
              print(f"GP fitting took {time.time() - start:.2f} seconds")
              start = time.time()
              # do EI optimization using LBFGS
              optei =  EIAcquisition(self.bounds,self.num_init_evals,self.seed,grad = self.gradcomb)
              self.x_next,y_bnext = optei.optimize_ei(gp=gp, y_target=min(-0.9,y_target), bounds = self.bounds, xi=0.0, num_restarts_ei=self.num_restarts_ei, num_init=5000)
              

              print("x_next:",self.x_next.shape)
              print(f"Optimizing EI took {time.time() - start:.2f} seconds")
              self.xnexts.append(self.x_next)
              #if self.gradcomb:
                  #self.plot(gp,optei,self.x,self.y[:,0],y_target,self.y[:,1:])
              #else:
                  #self.plot(gp,optei,self.x,self.y,y_target)

          except Exception:
              num_exceptions += 1
              if num_exceptions <= 10:
                  print("WARNING: Exception was raised, using a random point.")
                  # CHANGE
                  self.x_next = np.random.rand(self.lb)*(self.ubs-self.lbs)+self.lbs
                  raise Exception
              else:
                  raise RuntimeException("ERROR: Maximum number of exceptions raised!")


          del gp  # Free memory
          start = time.time()
          if self.gradcomb:
            fval,g = f(self.x_next,grad=True) 
            #checkgrad(fval,g,self.x_next)
            yvals.append(fval)
            gvals.append(g)
            self.y = np.array(yvals)
            gs = np.array(gvals)
            self.y = np.hstack((self.y.reshape(-1,1),gs))

            self.x = np.vstack((self.x, deepcopy(self.x_next[None, :])))
          else:
            fval = f(self.x_next,grad=False)  
            self.y = np.hstack((self.y, deepcopy(fval)))
            self.x = np.vstack((self.x, deepcopy(self.x_next[None, :])))
          if self.gradcomb:
                idx = np.argmin(self.y[:,0])
                self.bestysofar = self.y[idx,0]
          else:
                idx = np.argmin(self.y)
                self.bestysofar = self.y[idx]
                
          diff = np.abs(fval) - np.abs(self.bestP)
          print("diff",diff)
          if diff > 0.0001:
              self.bestP = np.abs(fval)      
          else:
              self.fixedcount +=1
          print("fixedcount, bestP , maxiter ",self.fixedcount, self.bestP, self.maxiter)
          if self.fixedcount == self.maxiter:
              self.loop = 0;

          print("bestysofar:",self.bestysofar, self.loop)

      return self.x_next



