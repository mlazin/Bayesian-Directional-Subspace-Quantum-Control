"""Quantum-control simulator and transition-probability gradient calculations."""

import numpy as np
import scipy
import scipy.io as sio
from scipy.linalg import null_space, solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs, spsolve

from sksparse.cholmod import cholesky


scipy.sparse.linalg.use_solver(useUmfpack=False)


def mysolve(A,b):
    if isinstance(A,csc_matrix):
        if b.ndim==2 and b.shape[1]==1:
            return spsolve(A,b[:,0])[:,None]
        if b.ndim==2 and b.shape[0]==1:
            return spsolve(A.T,b[0,:])[None,:]
        return spsolve(A,b)
    return solve(A,b)


class SQsim:
    def __init__(self,sfile, psi_init, psi_target):

        # Load data from the file

        mat= sio.loadmat(sfile)
        self.states = mat['states']
        self.H0 = mat['H0']
        self.mu = mat['mu']
        self.M = mat['M']
        self.B = mat['B']
        self.energy = mat['energy']
        
        # added -1 below (because of 0-versus-1 indexing)
        self.psi = self.states[:, psi_init-1, None]
        self.phi = self.states[:, psi_target-1, None]
        self.psi_in = psi_init-1
        self.psi_target = psi_target-1
        self.Nt = None
        self.tau = None
        self.fld = None
        self.system = None
        self.P = None
        self.dP = None


    def startup(self):
        E_0 = (self.states[:,0,None].T @ self.H0 @ self.states[:,0,None]).item()
        H0 = self.H0 - E_0 * self.M;
        qtheta = np.abs(np.arccos(np.abs(self.phi.T @ (self.M @ self.psi)))).item()

        E_i = (self.psi.T @ H0 @ self.psi).item()
        E_f = (self.phi.T @ H0 @ self.phi).item()
        QSL_heis = 2 * qtheta / np.abs(E_f - E_i)
        QSL_tamm = 2 * qtheta / np.abs(E_i + E_f)
        T_tot = 10 * QSL_heis               
        tau = QSL_tamm / 12
        Ntau = max(round(T_tot / tau), 720)
        tau = T_tot / Ntau
        tgrid = tau * np.arange(Ntau + 1)
        self.Nt = len(tgrid)
        self.tau = tau
        mu, new_axes = self.precond_rotate(self.psi, self.phi, self.mu[0,:], self.M)
        mu, new_scale = self.precond_scale(H0, mu, self.M)

        self.fld = {
                'axes': new_axes,
                'scale': new_scale,
                'size': (len(mu), len(tgrid)),
                'tgrid': tgrid,
                'actual2basis': new_scale,#lambda x: (new_axes.T @ x) / new_scale,
                'basis2actual': new_axes#lambda x: np.linalg.solve(new_axes.T, x * new_scale)
                }

        self.system = {
                'Npts': H0.shape[0],#len(H0),
                'Ndim': mu.shape[0],#len(mu),
                'H0': H0,
                'mu': mu,
                'M': self.M,
                'B': self.B,
                'psi': self.psi,
                'phi': self.phi,
                'n_i': self.psi_in,
                'n_f': self.psi_target,
                'energy':  self.energy
                }

        return self.system, self.fld

    def precond_rotate(self,psi, phi, mu, M):

        Ndim = mu.shape[0]  # Get the number of elements in mu
        tdm = np.zeros((Ndim, 1))  # Initialize tdm as a column vector

        for i in range(Ndim):
            # Perform matrix-vector multiplication
            product = mu[i] @ psi  # Result should be a 1D array

            # Ensure the product is 1D
            #product = np.squeeze(product)  # Remove any extra dimensions

            # Solve M \ product
            intermediate = mysolve(M, product)  # Solve M \ product

            # Compute abs(phi' * intermediate)
            tdm[i, 0] = abs((phi.T @ intermediate).item())  # Use np.dot for 1D arrays

        tdm = tdm / np.linalg.norm(tdm, 2)

        new_axes = np.column_stack((tdm, null_space(tdm.reshape(1, -1))))

        new_mu = np.zeros_like(mu)
        for jj in range(Ndim):
            for kk in range(Ndim):
                new_mu[jj] += new_axes[kk, jj] *mu[kk]

        return new_mu, new_axes

    def precond_scale(self,H0, mu, M=None):
        if M is None:
            M = np.eye(H0.shape[0])
        Ndim = mu.shape[0]  # Get the number of elements in mu

        # Calculate spectral bounds for H0

        reH, imH = self.spectralBounds(H0, M)
        H0_scl = np.abs(np.diff(reH + 1j * imH))
        H0_scl = max(H0_scl[0], np.finfo(float).eps)


        # Initialize scale factors
        scale_factor = np.ones(Ndim)
        for kk in range(Ndim):
            # Get spectral radius for mu[kk]
            reMu, imMu = self.spectralBounds(mu[kk], M)

            mu_scl = np.abs(np.diff(reMu + 1j * imMu))
            mu_scl = max(mu_scl, np.finfo(float).eps)

            scale_factor[kk] = H0_scl[0] / mu_scl[0,0] #single value , comes as array

        scaled_mu = mu.copy() 

        for kk in range(Ndim):
            scaled_mu[kk] *= scale_factor[kk] 

        return scaled_mu, scale_factor


    def spectralBounds(self,H, M=None):
        if M is None:
            M = np.eye(H.shape[0])

        # Check near-Hermitian property
        symm = self.nearHermitian(H, 1e-13)

        # Options
        # Decompose matrix into Hermitian and skew-Hermitian parts
        herm = (H + H.T) / 2
        skew = (H - H.T) / 2

        # Parameters
        tol = 1e-12
        maxit = 400
        p = 64  # Number of Lanczos vectors (ncv in SciPy)

        if symm == 0:
            eigenvalues_large, _ = eigs(herm, M=M, k=1, which="LR", tol=tol, maxiter=maxit, ncv=p)
            eigenvalues_small, _ = eigs(herm, M=M, k=1, which="LM", sigma=0, tol=tol, maxiter=maxit, ncv=p)
            realBnds = np.sort(np.real(np.concatenate([eigenvalues_large, eigenvalues_small]))).reshape(1, -1)

            eigenvalues_large, _ = eigs(skew, M=M, k=1, which="LR", tol=tol, maxiter=maxit, ncv=p)
            eigenvalues_small, _ = eigs(skew, M=M, k=1, which="LM", sigma=0, tol=tol, maxiter=maxit, ncv=p)
            imgBnds = np.sort(np.real(np.concatenate([eigenvalues_large, eigenvalues_small]))).reshape(1, -1)

        elif symm == +1:
            # Find the largest eigenvalue (real part)
            eigenvalues_large, _ = eigs(herm, M=M, k=1, which="LR", tol=tol, maxiter=maxit, ncv=p)
            eigenvalues_small, _ = eigs(herm, M=M, k=1, which="SR", tol=tol, maxiter=maxit, ncv=p)
                    
            """
            # To find the smallest eigenvalue, use the shift-invert method
            eigenvalues_small, _ = eigs(herm, M=M, k=1, which="LM", sigma=0, tol=tol, maxiter=maxit, ncv=p)
            if eigenvalues_small <  1e-12:
                eigenvalues_small = -eigenvalues_large
            """ 

            # Combine, sort, and reshape the real parts of the eigenvalues
            realBnds = np.sort(np.real(np.concatenate([eigenvalues_large, eigenvalues_small]))).reshape(1, -1)
            imgBnds = np.array([0,0])
        elif symm == -1:
            realBnds =np.array([0,0])
            eigenvalues_large, _ = eigs(skew, M=M, k=1, which="LR", tol=tol, maxiter=maxit, ncv=p)
            eigenvalues_small, _ = eigs(skew, M=M, k=1, which="LM", sigma=0, tol=tol, maxiter=maxit, ncv=p)
            imgBnds = np.sort(np.real(np.concatenate([eigenvalues_large, eigenvalues_small]))).reshape(1, -1)

        return realBnds, imgBnds


    def nearHermitian(self,A, tol=1e-12):

        if isinstance(A, csc_matrix):  # Ensure mu_i is sparse
            A = A.toarray()

        diff = A - A.conj().T
        sym_diff = np.linalg.norm(diff, ord='fro')

        if sym_diff < tol:
            return +1

        skew_diff = np.linalg.norm(A + A.conj().T, ord='fro')

        if skew_diff < tol:
            return -1

        if isinstance(A, np.ndarray) and A.size > 0:
            errorreal = np.spacing(A.real) + np.spacing(A.real.conj().T)
            errorimag = np.spacing(A.imag) + np.spacing(A.imag.conj().T)
            error = errorreal + errorimag
        else:
            error = tol / (1 + tol) * np.ones_like(A)

        A_herm = np.abs(A - A.conj().T) / 2 - error
        if np.all(A_herm[A_herm != 0] < 0):
            return +1

        A_skew = np.abs(A + A.conj().T) / 2 - error
        if np.all(A_skew[A_skew != 0] < 0):
            return -1

        return 0


    def prop_CN(self,psi, field, tau, sys):
        # Crank-Nicoloson propagator
        M = sys['M'] # overlap operator
        H0 = sys['H0']# static Hamiltonian
        mu = np.array(sys['mu'])#field operators

        # get Hamiltonian at endpoints
        Ht1 = H0.copy()
        Ht2 = H0.copy()

        for i in range(mu.shape[0]):
            Ht1 += field[i, 0] * mu[i]
            Ht2 += field[i, 1] * mu[i]

        Ht1, Ht2 = (Ht1 + Ht2) / 2, (Ht1 + Ht2) / 2

        is_bra = psi.shape[1] == M.shape[0]
        is_ket = psi.shape[0] == M.shape[1]

        if (is_bra and is_ket) or (not is_bra and not is_ket):
            raise ValueError("ERROR:the initial state psi has invalid shape for a bra/ket vector")
        elif is_bra:
            psi = mysolve(M + (1j * tau / 2) * Ht2, psi)
            psi = psi @ (M - (1j * tau / 2) * Ht1)
        elif is_ket:
            psi = (M - (1j * tau / 2) * Ht1) @ psi
            psi = mysolve(M + (1j * tau / 2) * Ht2, psi)
        return psi

    def propagate(self, psi, field, tgrid, propFunc, sys, ctransp=False, normalize=True):

        M = sys["M"]
        mu = np.array(sys["mu"])

        if field.shape[0] != len(mu):
            raise ValueError("ERROR: Dimension mismatch in field and mu.")
        elif field.shape[1] != len(tgrid):
            raise ValueError("ERROR: Dimension mismatch in field and tgrid.")


        tau = np.diff(tgrid)
        ntau = len(tau)

        if ctransp:
            field = np.flip(field, axis=1)
            tau = np.flip(tau)
            psi = psi.T

        is_bra = psi.shape[1] == M.shape[0]
        is_ket = psi.shape[0] == M.shape[1]

        if (is_bra and is_ket) or (not is_bra and not is_ket):
            raise ValueError("ERROR: State dimension mismatch with M.")

        elif is_bra:
            # added the "+1" below.  Although the MATLAB code allocates ntau
            # columns, the loop adds one more
            psi_prop = np.zeros((M.shape[0], ntau+1 ), dtype=np.complex_)
            Mt = M.T  # to make the mult in the "state_norm" below about 10x faster

            if normalize:
                state_norm = lambda x: np.abs(np.sum((Mt@x.T) * np.conj(x.T), axis=0))
                orig_norm = state_norm(psi)
            psi_prop[:,  ntau] =  psi[0,:]
            for jj in range(ntau - 1, -1, -1):
                psi = propFunc(psi, field[:, jj:jj + 2], tau[jj], sys)
                if normalize:
                    curr_norm = state_norm(psi)
                    psi *= np.sqrt(orig_norm / (curr_norm + (curr_norm == 0)))
                psi_prop[:, jj] = np.array(psi).flatten()
        elif is_ket:
            psi_prop = np.zeros((M.shape[0], ntau + 1), dtype=np.complex_)
            if normalize:
                state_norm = lambda x: np.abs(np.sum((M@x)*np.conj(x), axis=0)).item()
                orig_norm = state_norm(psi)
            psi_prop[:, 0] = psi[:,0]
            for jj in range(ntau):
                psi =propFunc(psi, field[:, jj:jj + 2], tau[jj], sys)
                if normalize:
                    curr_norm = state_norm(psi)
                    psi *= np.sqrt(orig_norm / (curr_norm + (curr_norm == 0)))
                psi_prop[:, jj + 1] = np.array(psi).flatten()

        if ctransp:
            psi_prop = np.transpose(psi_prop, axes=(1, 0, 2)).conj()

        return psi_prop

    def dProb(self,phiB, psifwd, tgrid, sys):
        M = sys['M']
        mu = sys['mu']
        Ntau = len(tgrid)
        Ndim = len(mu)

        dP = np.zeros((Ndim, Ntau))

        for i in range(Ndim):

            expmu = lambda a,b : np.dot(phiB[:,a],(mu[i]@ psifwd[:,b]))
            for j in range(1, Ntau - 1): #fixed Ntau-2 to Ntau-1 (Matlab ranges are inclusive)

                #overlap = np.dot(np.dot(psifwd[:, j].T, M), phiB[:, j].T)
                #overlap = psifwd[:,j]@ M @phiB[:,j]
                overlap = np.conj(psifwd[:,j]) @ M @ np.conj(phiB[:,j]) #must take complex conj
                    # because Matlab's ' operator is the complex conj transpose
                    # (not the transpose; .' is the transpose)
                if j == 1:
                    mu_prev = expmu(j-1, j-1 )
                    mu_curr = expmu(j, j)
                    mu_next = expmu(j + 1, j + 1)

                    mid_prev = expmu(j - 1, j) + expmu(j, j - 1)
                    mid_next = expmu(j + 1, j) + expmu(j, j + 1)
                else:
                    mu_prev = mu_curr
                    mu_curr = mu_next
                    mu_next = expmu(j + 1, j + 1)

                    mid_prev = mid_next
                    mid_next = expmu(j + 1, j) + expmu(j, j + 1)

                taut_prev = tgrid[j] - tgrid[j - 1]
                taut_next = tgrid[j + 1] - tgrid[j]

                dP_prev = taut_prev * (1/4 * (mu_curr + mu_prev) + 1/4 * mid_prev)
                dP_next = taut_next * (1/4 * (mu_curr + mu_next) + 1/4 * mid_next)



                dP[i,j] = np.imag(overlap * (dP_prev + dP_next))

        return dP

    def QC_simulate(self,field, tgrid, sys, computeGrad=False):
        psi = sys['psi']
        phi = sys['phi']
        M = sys['M']

        #  propagate the initial state
        psifwd= self.propagate(psi, field, tgrid, self.prop_CN, sys)

        psi_t = np.squeeze(psifwd)  # no effect, I think

        # find time-dependent transition probability
        Ntau = len(tgrid)

        Pt = np.zeros(Ntau)

        for i in range(Ntau):
            #Pt[i] = np.abs((phi.T  @ M @ psifwd[:,i]).item())
            
            Pt[i] = np.abs((phi.T  @ M @ psifwd[:,i]).item())**2

        P = Pt[-1]
        if computeGrad:
            phiB = self.propagate(phi.T @ M, field, tgrid, self.prop_CN, sys)

            if isinstance(M,csc_matrix):
                dM = cholesky(M.astype(phiB.dtype)) # need to convert M from real to complex
                    # to match phiB
                for j in range(phiB.shape[1]):
                    phiB[:,j] = dM(phiB[:, j])
            else:
                dM = scipy.linalg.cho_factor(M.todense())
                for j in range(phiB.shape[1]):
                    phiB[:,j] = scipy.linalg.cho_solve(dM,phiB[:,j])
                    
            """        
                    
            dP = self.dProb(phiB, psifwd, tgrid, sys)  # Compute gradient
            dP_mid = 0.5 * (dP[:, 1:-1] + dP[:, 0:-2])
            dP_area = np.sum(np.diff(tgrid[:int(dP_mid.shape[0])]) * np.sqrt(np.sum(dP_mid**2, axis=0)), axis=0)
            dP = dP * np.pi / (dP_area if dP_area != 0 else 1)
            """
            
            dP = self.dProb(phiB, psifwd, tgrid, sys)  # Compute gradient
            
        else:
            dP = []

        return P, Pt, psi_t, dP
