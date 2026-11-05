"""Gaussian-process Bayesian optimizer used by BO-DSO."""

import numpy as np
import torch
import gpytorch
from .GPyGP import MyGPGD, MyGPNOGD
from .EIacquisition import optimize_acquisition, expected_improvement, model_negmean

# bayesopt
class BayesOpt_wrapper:
    def __init__(self, bounds, mloop, grad, maxiter):
        self.bounds     = bounds          # search box for acquisition
        self.x          = 0               # design points (will become numpy / torch)
        self.y          = 0               # observations
        self.maxiter    = maxiter         # "patience" for no-improvement iters
        self.task       = 0               # #tasks for multitask GP (f + grads)
        self.grad       = grad            # True = use gradient-augmented GP
        self.mloop      = mloop           # GP training steps per BO iter
        self.loop       = 1               # outer loop flag
        self.fixedcount = 0               # number of consecutive non-improving iters
        self.bestP      = 0               # best |best_y| seen so far

    def run(self, f, initx, inity=None, initgrad=None, maxitt=2, searchdir=None, search0=None):
        """
        f      : objective; for grad=True, must return (fval, grad) for f(x, grad=True)
        initx  : initial design points, shape (n_init, d)
        inity  : optional initial y-values (rarely used in your case)
        maxitt : BO iterations per call
        searchdir : search only along this line
        search0: "origin" for searchdir
        """
        self.fixedcount = 0
        self.loop = 1

        self.x = initx
        print("x len.....", self.x.shape[1])

        if inity is not None:
            self.y = inity
            # CHANGE 1: part of change below
            gs = initgrad
        else:
            self.y = np.array([])

            # Initial evaluations
            ys = []
            gs = []
            for i in range(len(initx)):
                if self.grad:
                    # gradient-enabled: f should return (fval, g)
                    fval, g = f(initx[i], grad=True)  # minus sign handled inside f if needed
                    gs.append(g)
                else:
                    # non-gradient version: scalar only
                    fval = f(initx[i])
                ys.append(fval)

            self.y = np.array(ys)

        if self.grad:
            gs = np.array(gs)
            # y shape: (n_init, 1 + d)  -> columns: [f, grad_1, ..., grad_d]
            self.y = np.hstack((self.y.reshape(-1, 1), gs))

        # Initial data as torch tensors
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

        # BO main loop
        while self.loop > 0:
            print("Starting BO batch; maxitt =", maxitt)
            for i in range(maxitt):

                # Build GP model
                if self.grad:
                    # f + d gradient components
                    self.task = 1 + len(self.x[0])
                    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                        num_tasks=self.task,
                        has_task_noise = False,
                        has_global_noise = True,
                    )
                    model = MyGPGD(self.x, self.y, likelihood)
                else:
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    model = MyGPNOGD(self.x, self.y, likelihood)

                # Train GP
                model.train()
                likelihood.train()

                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
                model.covar_module.base_kernel.lengthscale[0] = 0.1
                model.likelihood.noise[0] = 1e-3
                print(model.likelihood.noise)
                model.likelihood.noise = 2e-4

                nkitt = int(self.mloop*self.x.shape[1])
                for k in range(nkitt):
                    optimizer.zero_grad()
                    output = model(self.x)
                    loss = -mll(output, self.y)
                    loss.backward()
                    if k%(nkitt//5) == 0:
                        #print(
                        #    'Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                        #        k + 1, self.maxiter, loss.item(),
                        #        model.covar_module.base_kernel.lengthscale.item(),
                        #        model.likelihood.noise.item()
                        #    )
                        #)
                        print(f'Iter {k+1} / {self.mloop*self.x.shape[1]} lengthscale: {model.covar_module.base_kernel.lengthscale.detach()}   noise: {model.likelihood.noise.detach()}')
                    optimizer.step()
                    model.likelihood.noise = 1e-4


                # Switch to numpy for acquisition & bookkeeping
                self.x = self.x.detach().numpy()
                self.y = self.y.detach().numpy()

                if self.grad:
                    # first column is f
                    best_y = self.y[:, 0].min()
                else:
                    best_y = self.y.min()

                if searchdir is not None:
                    class searchmodel:
                        def __init__(self,m,x0,xdir):
                            self.m = m
                            self.x0 = torch.from_numpy(x0)
                            self.xdir= torch.from_numpy(xdir)

                        def eval(self):
                            self.m.eval()

                        def __call__(self,x):
                            if self.x0.shape[0]>1:
                                return self.m(self.x0+x[:,None]*self.xdir)
                            return self.m(self.x0+x*self.xdir)

                    smodel = searchmodel(model,search0,searchdir)
                    b=np.array(f.bounds())
                    print('b',b)
                    print('searchdir',searchdir)
                    print('search0',search0)
                    bb = (b-search0)/searchdir
                    print('bb',bb)
                    mins = bb.min(axis=0)
                    maxs = bb.max(axis=0)
                    print(mins)
                    print(maxs)
                    sbounds = np.array([[mins.max(),maxs.min()]]).T
                    print('sbounds',sbounds)

                    x_next1,ei1 = optimize_acquisition(
                        expected_improvement,
                        smodel,
                        likelihood,
                        best_y,
                        sbounds,
                        grad=self.grad,
                        n=1,
                        retval=True
                    )
                    print('selected',x_next1,ei1)
                    if ei1<1e-10: # too little expected improvement,
                        # just go after the function itself
                        x_next1 = optimize_acquisition(
                            model_negmean,
                            smodel,
                            likelihood,
                            best_y,
                            sbounds,
                            grad=self.grad,
                            n=1,
                            plot=False,
                        )
                        print('reselected',x_next1)
                    x_next = search0+x_next1*searchdir
                else:
                    # Optimize acquisition (EI)
                    x_next = optimize_acquisition(
                        expected_improvement,
                        model,
                        likelihood,
                        best_y,
                        f.bounds(),
                        grad=self.grad,
                        n=self.x.shape[1],
                        xvals=self.x,yvals=self.y
                    )

                # Evaluate at new point
                if self.grad:
                    fval, g = f(x_next, grad=self.grad)
                else:
                    fval = f(x_next)

                print('----------')
                print('old x pts', self.x)
                print("old values fval", self.y)
                print('new x pt',x_next)
                print("new xnext fval", fval)
                print("besty", best_y)
                print('----------')

                fval = np.array([fval])
                if self.grad:
                    newY = np.hstack((fval, g))
                    self.y = np.vstack((self.y, newY.reshape(1, -1)))
                else:
                    self.y = np.hstack((self.y, fval))

                self.x = np.vstack((self.x, x_next[None, :]))

                # Back to torch for next iteration
                self.x = torch.from_numpy(self.x)
                self.y = torch.from_numpy(self.y)

                # Stopping logic based on |best_y|
                diff = np.abs(best_y) - self.bestP
                print("diff", diff)
                if diff > 0.0001:
                    self.bestP = np.abs(best_y)
                    self.fixedcount = 0   
                else:
                    self.fixedcount += 1

                print("fixedcount, bestP , maxiter ",
                      self.fixedcount, self.bestP, self.maxiter)

                if self.fixedcount == self.maxiter:
                    self.loop = 0
                    # break current batch early when converged
                    break

            # if we broke from inner loop, stop outer while too
            if self.loop == 0:
                break

        print("end of boopt ")
        if self.grad:
            besti = np.argmax(self.y[:,0])
            return self.x[besti], self.y[besti,0]
        else:
            besti = np.argmax(self.y)
            return self.x[besti], self.y[besti]
    
