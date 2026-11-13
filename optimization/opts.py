"""Optimization wrappers used by the public experiment runner.

This module contains only the optimizer classes required by run_experiment.py:
DirectionalAlphaWrapper1D, DirectionalAlphaWrapper, and scipyopt_linesearch.

"""

import numpy as np
from scipy.optimize import line_search

class DirectionalAlphaWrapper1D(object):
    def __init__(self,
                 baseopt,
                 sim,
                 alpha_bounds=(-0.05, 0.05),
                 max_outer_iters=5,
                 theta_expand=0.5): # from 0.5 to 0.1 
        
        self.baseopt = baseopt
        self.sim = sim
        self.alpha_bounds_arg = alpha_bounds
        self.max_outer_iters = int(max_outer_iters)
        self.total_budget = int(max_outer_iters)   # total budget on sim evals

        self.theta_expand = float(theta_expand)    # expansion threshold, theta_expand in [0,1]

        self.X0 = None
        self.D  = None
        self.alpha = None
        self.bestX = None
        self.bestY = None
        self.best_val = None

        # record storage
        self.recordX = []          # X evaluations
        self.recordY = []          # f(X)
        self.recordGrad = []       # gradients in X-space
        self.D_history = []        # snapshots of D for plotting
        self.alpha_eval_log  = []  # real alphas evaluated
        self.alpha_eval_vals = []  # matching objective values

        # counters
        self.dirlength = 1
        self.dircounter = 0
        self.expanditercount = []  # iterations where dim(D) increased
        self.inner_runs = 0


    @staticmethod
    def _safe_norm(v, eps=1e-12):
        n = float(np.sqrt(np.dot(v, v)))
        return max(n, eps)

    @staticmethod
    def _flatten(g):
        return np.asarray(g, float).reshape(-1)

    # internal bounds for u-space alpha
    def _au_bounds_vec(self, k):
        return np.vstack([np.full(k, -0.5), np.full(k, 0.5)])

    # affine map between internal a (u-space) and real alpha
    def _alpha_affine(self):
        lo, hi = self.alpha_bounds_arg            # e.g. (-0.05, 0.05)
        c = 0.5 * (float(lo) + float(hi))         # center of real interval
        s = (float(hi) - float(lo)) / (2 * 0.5)   # = hi - lo
        return c, s

    def _alpha_bounds_vec(self, k):
        lo, hi = self.alpha_bounds_arg if isinstance(self.alpha_bounds_arg, (tuple, list)) else (-0.5, 0.5)
        return np.vstack([np.full(k, float(lo)), np.full(k, float(hi))])

    def _orthogonalize_against_D(self, g):
        u = g.astype(float).copy()
        if self.D is not None and self.D.size:
            u -= self.D @ (self.D.T @ u)
        return u

    def _append_direction_from_gradient(self, gradX):
        g = self._flatten(gradX)
        print("g:", g)

        # debug plot:
        # plt.figure()
        # plt.bar(range(len(g)), g)
        # plt.title("Flattened gradient")
        # plt.show()

        u = self._orthogonalize_against_D(g)
        print("u:", u)
        nu = self._safe_norm(u)
        if nu < 1e-14:
            return False
        d_new = u / nu
        self.D = d_new[:, None] if (self.D is None or self.D.size == 0) else np.column_stack([self.D, d_new])
        return True

    def _make_phi(self, k):
        sim = self.sim
        D = self.D.copy()
        X0 = self.X0.copy()
        recordX, recordY, recordGrad = self.recordX, self.recordY, self.recordGrad
        flatten = self._flatten
        recordAlpha = self.alpha_eval_log
        recordAvals = self.alpha_eval_vals

        # rescaling alpha
        c, s = self._alpha_affine()          # real alpha = c + s * au
        bounds_au = self._au_bounds_vec(k)

        def phi(a, grad=True):
            # map internal a to real alpha, then to X
            a = np.asarray(a, float).reshape(-1)
            a_real = c + s * a               # real alpha
            X = X0 + D @ a_real

            if grad:
                val, gX = sim(X, grad=True)
                recordX.append(X)
                recordY.append(val)
                recordGrad.append(gX)        # store original gradient 

                recordAlpha.append(a_real.copy())
                recordAvals.append(float(val))

                # expand-count bookkeeping
                self.dircounter += 1
                temp = a_real.copy()
                print("temp", temp)
                if len(temp) != self.dirlength:
                    print("expanding dir...")
                    self.expanditercount.append(self.dircounter)
                    self.dirlength = int(len(temp))
                else:
                    print("not expanding dir...")

                # gradient transform for baseopt: df/dX -> df/d(alpha_real) -> df/du
                g_flat = flatten(gX)
                g_alpha_real = D.T @ g_flat      # df/d(alpha_real)
                g_au = s * g_alpha_real          # df/du (internal a)
                return float(val), g_au.reshape(-1)

            else:
                val = sim(X, grad=False)
                recordX.append(X)
                recordY.append(val)
                recordAlpha.append(a_real.copy())
                recordAvals.append(float(val))
                return float(val)

        phi.bounds = lambda: bounds_au
        return phi

    def set_anchor(self, X0):
        self.X0 = np.asarray(X0, float).reshape(-1)

    def rawgrad(self, f, initx):
        p, g = f(initx, grad=True)
        return p, g
        
    # Decide whether to expand the subspace using the given gradient.
    # if overlap < theta_expand - expand subspace with this gradient
    # else - keep exploring current plane
    #If we have no D yet - always expand.
    def _should_expand_with_gradient(self, grad):
    
        if self.D is None or self.D.size == 0:
            self.last_proj = 0.0
            return True
    
        g = self._flatten(grad)
        g = g / self._safe_norm(g)    # normalized gradient
    
        proj = self.D.T @ g           # projection onto current subspace
        proj_norm_sq = float(proj @ proj)
    
        #self.last_proj = proj_norm_sq
        print(f"Projection norm^2 = {proj_norm_sq:.6f}")
    
        return proj_norm_sq < self.theta_expand

    def run1d(self, f=None, xinit=None, iter=10):
        if f is not None:
            self.sim = f
        if self.X0 is None:
            raise ValueError("Please set the anchor via set_anchor(X0) first.")

        ambient_dim = int(self.X0.size)

        # Evaluate anchor
        X_cur = self.X0.copy()
        val_cur, gX = self.sim(X_cur, grad=True)
        self.bestX = X_cur.copy()
        self.best_val = float(val_cur)

        searchdir = gX / self._safe_norm(gX)
        search0 = self.bestX


        # record first point
        self.recordX = [X_cur]
        self.recordY = [val_cur]
        self.recordGrad = [gX]

        # First direction from anchor gradient
        ok = self._append_direction_from_gradient(gX)
        if not ok:
            raise RuntimeError("Couldn't create first direction: projected gradient ~ 0 at X0.")

        # record the initial basis
        self.D_history.append(self.D.copy())
        self.alpha = np.zeros(self.D.shape[1], float)

        # Outer loop
        while len(self.recordX) < self.total_budget:
            k = self.D.shape[1]

            # Build phi in current k-dimensional subspace
            phi = self._make_phi(k)

            # project recorded X back to alphas (real and internal)
            c, s = self._alpha_affine()
            alphas_all_real = [self.D.T @ (X - self.X0) for X in self.recordX]
            initx_k = np.array([(a - c) / s for a in alphas_all_real], float)
            inity_k = np.array(self.recordY, float).reshape(-1, 1)

            grads_alpha_real = np.array(
                [self.D.T @ self._flatten(g) for g in self.recordGrad],
                float
            )
            initgrad_k = s * grads_alpha_real

            sdirproj = self.D.T@searchdir
            sdir = sdirproj / self._safe_norm(sdirproj)
            _,_ = self.baseopt.run(phi,
                                   initx=initx_k,
                                   inity=inity_k,
                                   initgrad=initgrad_k,
                                   searchdir=sdir,
                                   search0=(self.D.T@(search0-self.X0) - c) / s,
                                   )

            # Pick global best among ALL recorded points
            i = int(np.argmin(self.recordY))
            self.bestX = self.recordX[i]
            self.best_val = float(self.recordY[i])
            bestgrad = self.recordGrad[i]
            bestgradnorm = bestgrad / self._safe_norm(bestgrad)
            proj = self.D.T @ bestgradnorm
            proj_norm = np.sqrt(float(proj @ proj))
            sdir_norm = sdir.dot(proj)
            print('proj dots',sdir_norm,proj_norm)
            if np.abs(sdir_norm)>0.5:
                search0 = self.bestX # ???
                print('continuing in this direction')
            elif proj_norm>0.5:
                print('changing direction, but keeping subspace')
                searchdir = bestgradnorm
                search0 = self.bestX
            else:
                print('changing direction and adding dimension to subspace')
                if self._append_direction_from_gradient(bestgradnorm):
                    self.D_history.append(self.D.copy())
                    self.alpha = np.append(self.alpha, 0.0)
                else:
                    print('failed to add new dimension to subspace')
                searchdir = bestgradnorm
                search0 = self.bestX

        i = int(np.argmin(self.recordY))
        return self.recordX[i], float(self.recordY[i])


    def run(self, f=None, xinit=None, iter=10):
        if f is not None:
            self.sim = f
        if self.X0 is None:
            raise ValueError("Please set the anchor via set_anchor(X0) first.")

        ambient_dim = int(self.X0.size)

        # Evaluate anchor
        X_cur = self.X0.copy()
        val_cur, gX = self.sim(X_cur, grad=True)
        print("E zero 1st grad from daw", gX)
        self.bestX = X_cur.copy()
        self.best_val = float(val_cur)

        # record first point
        self.recordX = [X_cur]
        self.recordY = [val_cur]
        self.recordGrad = [gX]

        # First direction from anchor gradient
        ok = self._append_direction_from_gradient(gX)
        if not ok:
            raise RuntimeError("Couldn't create first direction: projected gradient ~ 0 at X0.")

        # record the initial basis
        self.D_history.append(self.D.copy())
        self.alpha = np.zeros(self.D.shape[1], float)

        # Outer loop
        while len(self.recordX) < self.total_budget and self.D.shape[1] < ambient_dim:
            k = self.D.shape[1]

            # Build phi in current k-dimensional subspace
            phi = self._make_phi(k)

            # project recorded X back to alphas (real and internal)
            c, s = self._alpha_affine()
            alphas_all_real = [self.D.T @ (X - self.X0) for X in self.recordX]
            initx_k = np.array([(a - c) / s for a in alphas_all_real], float)
            inity_k = np.array(self.recordY, float).reshape(-1, 1)

            grads_alpha_real = np.array(
                [self.D.T @ self._flatten(g) for g in self.recordGrad],
                float
            )
            initgrad_k = s * grads_alpha_real

            # run inner optimizer
            # CHANGE 3: run returns two things and they are the best it
            #  found, but we are ignoring them
            _,_ = self.baseopt.run(phi,
                                   initx=initx_k,
                                   inity=inity_k,
                                   initgrad=initgrad_k)
            #bestX_full = np.asarray(bestX_full, float).reshape(-1)

            # Pick global best among ALL recorded points
            i = int(np.argmin(self.recordY))
            self.bestX = self.recordX[i]
            self.best_val = float(self.recordY[i])
            bestgrad = self.recordGrad[i]
            print('bestgrad sq mag:',bestgrad.dot(bestgrad))
            proggrad = self.D.T@ bestgrad
            print('bestgrad proj sq mag:',proggrad.dot(proggrad))

            # try to add one new direction each iteration
            if self.D.shape[1] >= ambient_dim:
                break

            added = False

            # decide expansion based on projection of best gradient
            if self._should_expand_with_gradient(bestgrad):
                if self._append_direction_from_gradient(bestgrad):
                    self.D_history.append(self.D.copy())
                    self.alpha = np.append(self.alpha, 0.0)
                    added = True
                    print(f"Expanded subspace: projection below threshold {self.theta_expand}")
                else:
                    print("Gradient is already inside the current subspace; no new direction added.")
            else:
                print(f"Skipping expansion: gradient mostly in current subspace (threshold = {self.theta_expand})")
                added = False
            
            if not added:
                print("No new direction found")

        i = int(np.argmin(self.recordY))
        return self.recordX[i], float(self.recordY[i])

    def get_history(self):

        X_arr = np.asarray(self.recordX, float)
        Y_arr = np.asarray(self.recordY, float)
        G_arr = np.asarray(self.recordGrad, dtype=object)
        expanditer = np.asarray(self.expanditercount, int)
        return X_arr, Y_arr, G_arr, self.D_history, expanditer


class DirectionalAlphaWrapper(object):
    def __init__(self,
                 baseopt,
                 sim,
                 alpha_bounds=(-0.05, 0.05),
                 max_outer_iters=5,
                 theta_expand=0.5):
        
        self.baseopt = baseopt
        self.sim = sim
        self.alpha_bounds_arg = alpha_bounds
        self.max_outer_iters = int(max_outer_iters)
        self.total_budget = int(max_outer_iters)   # total budget on sim evals

        self.theta_expand = float(theta_expand)    # expansion threshold, theta_expand in [0,1]

        self.X0 = None
        self.D  = None
        self.alpha = None
        self.bestX = None
        self.best_val = None

        # record storage
        self.recordX = []          # X evaluations
        self.recordY = []          # f(X)
        self.recordGrad = []       # gradients in X-space
        self.D_history = []        # snapshots of D for plotting
        self.alpha_eval_log  = []  # real alphas evaluated
        self.alpha_eval_vals = []  # matching objective values

        # counters
        self.dirlength = 1
        self.dircounter = 0
        self.expanditercount = []  # iterations where dim(D) increased
        self.inner_runs = 0


    @staticmethod
    def _safe_norm(v, eps=1e-12):
        n = float(np.sqrt(np.dot(v, v)))
        return max(n, eps)

    @staticmethod
    def _flatten(g):
        return np.asarray(g, float).reshape(-1)

    # internal bounds for u-space alpha
    def _au_bounds_vec(self, k):
        return np.vstack([np.full(k, -0.5), np.full(k, 0.5)])

    # affine map between internal a (u-space) and real alpha
    def _alpha_affine(self):
        lo, hi = self.alpha_bounds_arg            # e.g. (-0.05, 0.05)
        c = 0.5 * (float(lo) + float(hi))         # center of real interval
        s = (float(hi) - float(lo)) / (2 * 0.5)   # = hi - lo
        return c, s

    def _alpha_bounds_vec(self, k):
        lo, hi = self.alpha_bounds_arg if isinstance(self.alpha_bounds_arg, (tuple, list)) else (-0.5, 0.5)
        return np.vstack([np.full(k, float(lo)), np.full(k, float(hi))])

    def _orthogonalize_against_D(self, g):
        u = g.astype(float).copy()
        if self.D is not None and self.D.size:
            u -= self.D @ (self.D.T @ u)
        return u

    def _append_direction_from_gradient(self, gradX):
        g = self._flatten(gradX)
        print("g:", g)

        # debug plot:
        # plt.figure()
        # plt.bar(range(len(g)), g)
        # plt.title("Flattened gradient")
        # plt.show()

        u = self._orthogonalize_against_D(g)
        print("u:", u)
        nu = self._safe_norm(u)
        if nu < 1e-14:
            return False
        d_new = u / nu
        self.D = d_new[:, None] if (self.D is None or self.D.size == 0) else np.column_stack([self.D, d_new])
        return True

    def _make_phi(self, k):
        sim = self.sim
        D = self.D.copy()
        X0 = self.X0.copy()
        recordX, recordY, recordGrad = self.recordX, self.recordY, self.recordGrad
        flatten = self._flatten
        recordAlpha = self.alpha_eval_log
        recordAvals = self.alpha_eval_vals

        # rescaling alpha
        c, s = self._alpha_affine()          # real alpha = c + s * au
        bounds_au = self._au_bounds_vec(k)

        def phi(a, grad=True):
            # map internal a to real alpha, then to X
            a = np.asarray(a, float).reshape(-1)
            a_real = c + s * a               # real alpha
            X = X0 + D @ a_real

            if grad:
                val, gX = sim(X, grad=True)
                recordX.append(X)
                recordY.append(val)
                recordGrad.append(gX)        # store original gradient 

                recordAlpha.append(a_real.copy())
                recordAvals.append(float(val))

                # expand-count bookkeeping
                self.dircounter += 1
                temp = a_real.copy()
                print("temp", temp)
                if len(temp) != self.dirlength:
                    print("expanding dir...")
                    self.expanditercount.append(self.dircounter)
                    self.dirlength = int(len(temp))
                else:
                    print("not expanding dir...")

                # gradient transform for baseopt: df/dX -> df/d(alpha_real) -> df/du
                g_flat = flatten(gX)
                g_alpha_real = D.T @ g_flat      # df/d(alpha_real)
                g_au = s * g_alpha_real          # df/du (internal a)
                return float(val), g_au.reshape(-1)

            else:
                val = sim(X, grad=False)
                recordX.append(X)
                recordY.append(val)
                recordAlpha.append(a_real.copy())
                recordAvals.append(float(val))
                return float(val)

        phi.bounds = lambda: bounds_au
        return phi

    def set_anchor(self, X0):
        self.X0 = np.asarray(X0, float).reshape(-1)

    def rawgrad(self, f, initx):
        p, g = f(initx, grad=True)
        return p, g
        
    # Decide whether to expand the subspace using the given gradient.
    # if overlap < theta_expand - expand subspace with this gradient
    # else - keep exploring current plane
    #If we have no D yet - always expand.
    def _should_expand_with_gradient(self, grad):
    
        if self.D is None or self.D.size == 0:
            self.last_proj = 0.0
            return True
    
        g = self._flatten(grad)
        g = g / self._safe_norm(g)    # normalized gradient
    
        proj = self.D.T @ g           # projection onto current subspace
        proj_norm_sq = float(proj @ proj)
    
        #self.last_proj = proj_norm_sq
        print(f"Projection norm^2 = {proj_norm_sq:.6f}")
    
        return proj_norm_sq < self.theta_expand


    def run(self, f=None, xinit=None, iter=10):
        if f is not None:
            self.sim = f
        if self.X0 is None:
            raise ValueError("Please set the anchor via set_anchor(X0) first.")

        ambient_dim = int(self.X0.size)

        # Evaluate anchor
        X_cur = self.X0.copy()
        val_cur, gX = self.sim(X_cur, grad=True)
        print("E zero 1st grad from daw", gX)
        self.bestX = X_cur.copy()
        self.best_val = float(val_cur)

        # record first point
        self.recordX = [X_cur]
        self.recordY = [val_cur]
        self.recordGrad = [gX]

        # First direction from anchor gradient
        ok = self._append_direction_from_gradient(gX)
        if not ok:
            raise RuntimeError("Couldn't create first direction: projected gradient ~ 0 at X0.")

        # record the initial basis
        self.D_history.append(self.D.copy())
        self.alpha = np.zeros(self.D.shape[1], float)

        # Outer loop
        while len(self.recordX) < self.total_budget and self.D.shape[1] < ambient_dim:
            k = self.D.shape[1]

            # Build phi in current k-dimensional subspace
            phi = self._make_phi(k)

            # project recorded X back to alphas (real and internal)
            c, s = self._alpha_affine()
            alphas_all_real = [self.D.T @ (X - self.X0) for X in self.recordX]
            initx_k = np.array([(a - c) / s for a in alphas_all_real], float)
            inity_k = np.array(self.recordY, float).reshape(-1, 1)

            grads_alpha_real = np.array(
                [self.D.T @ self._flatten(g) for g in self.recordGrad],
                float
            )
            initgrad_k = s * grads_alpha_real

            # run inner optimizer
            bestX_full = self.baseopt.run(phi,
                                          initx=initx_k,
                                          inity=inity_k,
                                          initgrad=initgrad_k)
            bestX_full = np.asarray(bestX_full, float).reshape(-1)

            # Pick global best among ALL recorded points
            i = int(np.argmin(self.recordY))
            self.bestX = self.recordX[i]
            self.best_val = float(self.recordY[i])
            bestgrad = self.recordGrad[i]

            # try to add one new direction each iteration
            if self.D.shape[1] >= ambient_dim:
                break

            added = False

            # decide expansion based on projection of best gradient
            if self._should_expand_with_gradient(bestgrad):
                if self._append_direction_from_gradient(bestgrad):
                    self.D_history.append(self.D.copy())
                    self.alpha = np.append(self.alpha, 0.0)
                    added = True
                    print(f"Expanded subspace: projection below threshold {self.theta_expand}")
                else:
                    print("Gradient is already inside the current subspace; no new direction added.")
            else:
                print(f"Skipping expansion: gradient mostly in current subspace (threshold = {self.theta_expand})")
                added = False
            
            if not added:
                print("No new direction found")



        return self.bestX

    def get_history(self):

        X_arr = np.asarray(self.recordX, float)
        Y_arr = np.asarray(self.recordY, float)
        G_arr = np.asarray(self.recordGrad, dtype=object)
        expanditer = np.asarray(self.expanditercount, int)
        return X_arr, Y_arr, G_arr, self.D_history, expanditer


class scipyopt_linesearch:
    def __init__(self, bounds=None, options={}):
        self.method = 'line-search'
        self.usegrad = True
        self.options = options
        self.iter = self.options.get('maxfun', None)
        self.gds = []      # per-iteration logs: {'iter','x','f','grad_norm'}
        self.dirs = []     # descent directions per iteration (p_k)
        self.raw_grads = []    # NEW: store raw gradients at x_k (for plotting/checks)
        self.normalize_dirs = bool(self.options.get('normalize_dirs', True))  # NEW: default True


        if isinstance(bounds, tuple):
            self.radius_bounds = bounds[1]
            bounds_array = bounds[0]
            self.bounds = list(zip(bounds_array[0], bounds_array[1]))
            print("bounds length from scipyopt_linesearch", len(self.bounds))
        elif bounds is None:
            self.radius_bounds = None
            self.bounds = None
        else:
            self.radius_bounds = None
            self.bounds = list(bounds)

        if self.bounds is not None:
            self.low  = np.array([b[0] for b in self.bounds])
            self.high = np.array([b[1] for b in self.bounds])
        else:
            self.low = self.high = None

        self.res_list = []
        
    # make grad 1-D
    @staticmethod
    def _flatten(g):
        return np.asarray(g, float).reshape(-1)
        
    def _append_direction_from_gradient(self, gradX):
        g = self._flatten(gradX)
        print("g:",g)
        
        #plt.figure()
        #plt.bar(range(len(g)), g)
        #plt.title("Flattened gradient ls")
        #plt.show()
    
    def rawgrad(self, f, initx):
      # start point (last row if 2D)
      x = initx[-1] if getattr(initx, "ndim", 1) > 1 else initx
      x = x.astype(float).copy()
      p,g = f(x,grad = True)
      return p,g

    def run(self, f, initx, inity=None, initgrad=None, maxitt=10):
        self.res_list = []
        self.gds = []
        self.dirs = []
        self.raw_grads = []   # NEW

        # start point (last row if 2D)
        x = initx[-1] if getattr(initx, "ndim", 1) > 1 else initx
        x = x.astype(float).copy()

        # one eval helper with penalty + flatten
        def evalf(xx):
            val, grad = f(xx, grad=True)
            print("E zero 1st grad from la",grad)
            self._append_direction_from_gradient(grad)
            if hasattr(grad, "shape") and len(grad.shape) > 1:
                grad = grad.flatten()
            
            # spherical penalty 
            if self.radius_bounds is not None:
                excess  = np.sum(xx**2) - self.radius_bounds**2
                penalty = 10 * max(0, excess)
                comp = 0 if excess < 0 else 2 * xx
                val  = val  + penalty
                grad = grad + 10 * comp
            
            return val, grad

        # initial evaluation (counts toward budget) 
        fx, gx = evalf(x)
        self.raw_grads.append(gx.copy())  # NEW
        #maxitt -= 1  # we just evaluated once
        self.gds.append({'iter': 0, 'x': x.copy(), 'f': float(fx), 'grad_norm': float(np.linalg.norm(gx))})

        # iterate until eval budget exhausted
        it = 0
        while maxitt > 0:
            it += 1

            # direction = -gradient
            #p = -gx
            #self.dirs.append(p.copy())
            
            # direction = -gradient (optionally normalized)
            p = -gx.copy()
            if self.normalize_dirs:                       
                n = np.linalg.norm(p)
                if n > 0:
                    p /= n

            self.dirs.append(p.copy())
            
            # simple SciPy line search along p
            ls = line_search(lambda z: evalf(z)[0],
                             lambda z: evalf(z)[1],
                             x, p, gx, fx)
            alpha = ls[0] if ls[0] is not None else 1e-2  # fallback small step
            fc    = ls[1] if ls[1] is not None else 0     # function evals inside LS
            gc    = ls[2] if ls[2] is not None else 0     # grad evals inside LS

            # take step
            x = x + alpha * p


            # post-step evaluation (counts as 1)
            fx, gx = evalf(x)
            self.raw_grads.append(gx.copy())  # NEW
            self.gds.append({'iter': it, 'x': x.copy(), 'f': float(fx), 'grad_norm': float(np.linalg.norm(gx))})

            # subtract budget used this iteration: line-search (fc+gc) + our post-step eval (1)
            used = fc + gc + 1
            maxitt -= used
            # optional: stop if budget is gone
            if maxitt <= 0:
                break

        # record SciPy-like result
        res = {'x': x, 'fun': fx, 'nit': it}
        self.res_list.append(res)
        return res['x'], res['fun']
