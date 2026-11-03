"""Objective wrappers for the quantum-control simulator."""

import numpy as np
import scipy

from .qsimnp import SQsim


class Sqfun:
    def __init__(self,sfile,psi_init,psi_target,bound,device = 'cpu', flatten_grad=True): 
       
        self.psi_init = psi_init
        self.psi_target = psi_target
        self.sims =  [SQsim(sfile,psi_init, psi_target)]
        for sim in self.sims:
          sim.startup()
        self.sfile = sfile
        #self.nsteps = nsteps
        self.counter = 0
        self.sys = None
        self.maxe = bound
        self.E = None                      #change
        self.flatten_grad = flatten_grad   #change

    def __call__(self,x,fid=0,grad=False):
        
        sim = self.sims[int(fid)]
        Ex = x[:sim.Nt]
        Ey = x[sim.Nt:]
        E = np.array([Ex,Ey])
        self.sys = sim.system
        fld = sim.fld
        tg = fld['tgrid']
        self.E = E

        if not grad:
            P,pt,psit,dP = sim.QC_simulate(E,tg,self.sys,False)
            return -P
        else:
           P,pt,psit,dP = sim.QC_simulate(E,tg,self.sys,True)
           g = dP.flatten() if self.flatten_grad else dP           #returns 2×Nt gradient or a flattened gradient
           return -P, -g
           #return -P, -dP


    #7-13: 7-8,7-9,7-10,7-11,7-12,7-13
    def getenergy(self,fid=0):
       energy = self.sims[fid].energy
       energy = energy.flatten()
       energies = []
       diff = self.psi_target - self.psi_init
       for i in range(diff):
         energies.append(np.abs(energy[self.psi_init]- energy[self.psi_init+i+1]))
         if self.psi_target!=self.psi_init+i+1:
           energies.append(np.abs(energy[self.psi_init+i+1]-energy[self.psi_target]))
       return energies
   
    def bounds(self):
        minbnds = np.hstack((np.ones((1,self.sims[0].Nt))*-self.maxe[0],np.ones((1,self.sims[0].Nt))*-self.maxe[1]))
        maxbnds = np.hstack((np.ones((1,self.sims[0].Nt))*self.maxe[0],np.ones((1,self.sims[0].Nt))*self.maxe[1]))
        return np.vstack((minbnds,maxbnds))


class testf:
    def __init__(self,nt=10):
        self.nt = nt

    def bounds(self):
        return np.vstack((-np.ones((1,2*self.nt)),
                           np.ones((1,2*self.nt))))

    def __call__(self,x,fid=0,grad=False):
        inx = x[:self.nt]
        iny = x[self.nt:]

        fval = ((inx-np.linspace(-1,1,self.nt))**2).sum() + (iny**2).sum()
        if not grad:
            return fval
        else:
            inxgrad = 2*(inx-np.linspace(-1,1,self.nt))
            inygrad = 2*iny
            return fval, np.hstack((inxgrad,inygrad))

class freqdomain:
    # nfreq does not include the DC component
    # so there will be a total of (2*freq+1)*2
    # parameters (first 2 because of Real&Img components,
    # second 2 because of Ex and Ey)
    # !!if nfreq is None, this uses all of the frequencies
    def __init__(self,origfn,nfreq=None): #6.change
        self.f = origfn
        self.penalty = 10
        self.nt = origfn.bounds().shape[1]//2
        #print("checking the nt...... from freqdomain....",self.nt)
        self.nfullfreq = self.nt//2+1 if self.nt%2==0 else (self.nt+1)//2

        if nfreq is not None:
            self.nfreq = nfreq
        else:
            self.nfreq = self.nfullfreq-1
            
        self.E = origfn.E


    def bounds(self):
        maxxb = self.f.bounds()[1,0]
        maxyb = self.f.bounds()[1,-1]
        nf = self.nfreq*2+1
        
        minbnds = np.hstack((np.ones((1,nf))*-maxxb,np.ones((1,nf))*-maxyb))
        maxbnds = np.hstack((np.ones((1,nf))*maxxb,np.ones((1,nf))*maxyb))
        return (np.vstack((minbnds,maxbnds))*self.nt/2, np.minimum(maxxb,maxyb)*nf/2)
        
        #return self.f.bounds()


    def coupleddims(self,i):
        n1 = self.nfreq+1
        n2 = n1+self.nfreq
        if i==0 or i==n2:
            return []
        if i<n1:
            return [i+self.nfreq]
        if i<n2:
            return [i-self.nfreq]
        if i<n2+n1:
            return [i+self.nfreq]
        else:
            return [i-self.nfreq]

    def __call__(self,xorig,fid=0,grad=False):
        #print(list(enumerate(xorig)))
        x,xmag,ymag = self.map(xorig)
        
        if not grad:
            #xb = self.f.bounds()[0][1,0]*self.nt/2
            #yb = self.f.bounds()[0][1,-1]*self.nt/2
            rawf = self.f(x,fid=fid,grad=grad)
            ret = rawf #+ self.penalty * (np.maximum(0,xmag-xb) + np.maximum(0,ymag-yb))
            #print(f'{ret} = {rawf} + {self.penalty} * (np.maximum(0,{xmag-xb}) + np.maximum(0,{ymag-yb})')
            return ret
        else:
            fval,gval = self.f(x,fid=fid,grad=grad)
            gval = self.gmap(gval)

            if False: # this was to test grad by finite difference
                delx = 1e-10
                def addtoi(x,i,v):
                #    print('adding to ',i)
                    ret = x.copy()
                    ret[i] += v
                    return self.map(ret)
                def calcdiff(f1,f2,delx):
                #    print(f1,f2,delx)
                #    print((f1-f2)/delx)
                    return (f1-f2)/delx
                n = 10 #len(x)
                fdgrad = np.array([
                    calcdiff(self.f(addtoi(xorig,i,delx),fid=fid,grad=False),fval,delx)
                    for i in range(n)])
                #print('g val:',gval[:n])
                #print('fd vl:',fdgrad)
            self.E = self.f.E
            return fval,gval


    def freqtofeat(self,freq):
       n1 = self.nfreq+1
       n2 = n1 +self.nfreq
       ds = set()
       for i in range(len(freq)): # freq comes as list of energy differences

          #E = np.sin(freq[i]*np.arange(self.nt))
          E = np.sin(freq[i]*self.f.sims[0].fld['tgrid'])
          Efreq = scipy.fft.rfft(E,n = self.nfullfreq)
          d = np.argmax(Efreq)
          ds |= set([d]+self.coupleddims(d)+[d+n2]+self.coupleddims(d+n2))
       #d = np.argmax(np.array(ds))
       #print("d couple d , d+n2 and couple d+n2", d, self.coupleddims(d), d+n2, self.coupleddims(d+n2))
       #return [d] + self.coupleddims(d) + [d+n2] + self.coupleddims(d+n2)
       #exit(1)
       return list(ds)

    # just as the algo
    def freqtofeat_diff(self,freq):
       n1 = self.nfreq+1
       n2 = n1 +self.nfreq
       fks = []  # for fk from the algorithm f1 a-b, f2 b-c f3 c-d and so on so k is psi_target - psi_init so 9 to 13 that is there are 4 fs
       #freq all energy diff
       for i in range(len(freq)):
          E = np.sin(freq[i]*np.arange(self.nt))
          #print("shape of E from freqtofeat",E.shape)
          f = scipy.fft.rfft(E)
          #print("shape of f...",f.shape,f)
          fks.append(np.argmax(f))
          #fks.append(f)
       d = np.argmax(np.array(fks))
       #print("the d from freqtofeat",d,fks)
       #print("d couple d , d+n2 and couple d+n2", d, self.coupleddims(d), d+n2, self.coupleddims(d+n2))
       return [d] + self.coupleddims(d)  + [d+n2] + self.coupleddims(d+n2)

    def getdimsfromfreq(self):
      freq = self.f.getenergy()
      #print("the freqs from getdimsfromfreq.....",freq)
      return self.freqtofeat(freq)

    def map(self,x):
        
        xdim = int(len(x))
        xdim = int(xdim/2)
        
        # CHANGE3
        xdim = self.nt

        Exfreq = np.zeros(self.nfullfreq,dtype=complex)
        Eyfreq = np.zeros(self.nfullfreq,dtype=complex)
        n1 = self.nfreq+1
        n2 = n1+self.nfreq
        Exfreq[0] = x[0]
        Exfreq[1:n1] = x[1:n1] + 1j*x[n1:n2]
        Exabs = np.absolute(Exfreq)
        #Exfreq = Exfreq*np.max(Exabs)/np.sum(Exabs)

        Eyfreq[0] = x[n2]
        Eyfreq[1:n1] = x[n2+1:n2+n1] + 1j*x[n2+n1:]
        Eyabs = np.absolute(Eyfreq)
        #Eyfreq = Eyfreq*np.max(Eyabs)/np.sum(Eyabs)

        #if xdim % 2 ==0:
        #  Ex = scipy.fft.irfft(Exfreq)
        #  Ey = scipy.fft.irfft(Eyfreq)
        #else:
        #  Ex = scipy.fft.irfft(Exfreq,n=xdim)
        #  Ey = scipy.fft.irfft(Eyfreq,n=xdim)
        """
        #6_18_2025, Change for Bin 4
        Exfreq = np.zeros(self.nfullfreq,dtype=complex)
        Eyfreq = np.zeros(self.nfullfreq,dtype=complex)
        #Exfreq[4] = 0-1j*x[0]      #x[0] = 0.54075
        Exfreq[4] = x[0]      #x[0] = 0.54075
        """
        # CHANGE3
        Ex = scipy.fft.irfft(Exfreq,n=xdim)
        Ey = scipy.fft.irfft(Eyfreq,n=xdim)


        return np.hstack((Ex,Ey)), np.sum(Exabs), np.sum(Eyabs)

    def gmap(self,g):
        if len(g.shape)>1:
          g = g.flatten()

        m = g.shape[0]//2

        gfreqx = scipy.fft.rfft(g[:m])
        gfreqy = scipy.fft.rfft(g[m:])
        realx = np.real(gfreqx[:self.nfreq+1])
        imgx = np.imag(gfreqx[1:self.nfreq+1])
        realy = np.real(gfreqy[:self.nfreq+1])
        imgy = np.imag(gfreqy[1:self.nfreq+1])
        ret = np.hstack((realx,imgx,realy,imgy))*2/m
        ret[0]/=2
        ret[2*self.nfreq+1]/=2
        return ret
