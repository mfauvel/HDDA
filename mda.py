# -*- coding: utf-8 -*-
import scipy as sp
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
import hdda as HDDA

## Worker function for MDA
def workerMda(x,MODEL,th,C):
    """
    """
    model = HDGMM()
    model.bic = HDDA.MIN
    iter = 0
    while model.bic == HDDA.MIN:
        model.fit_all(x,MODEL=MODEL,th=th,C=C,random_state=iter)
        iter+=1
    return model

class MDA():
    def __init__(self):
        """
        """
        self.model = []
        self.prop = []
        self.j = []

    def fit(self,x,y,MODEL=['M1','M2','M3','M4','M5','M6','M7','M8'],th=[0.1], C = [1,2,3,4,5,6]):
        """
        """
        n,d = x.shape
        nbC = int(y.max())

        # Get the indices of samples for each class
        self.j = [sp.where(y==(c_+1))[0] for c_ in xrange(nbC)]

        # Get the proportion of each class
        self.prop = [float(j_.size)/n for j_ in self.j]

        # Learn each class mixture
        self.model = Parallel(n_jobs=4)(delayed(workerMda)(x[j_,:],MODEL,th,C) for j_ in self.j) # TODO: tester si cela n'a pas convergé -> il faut relancer !!

        

    def predict(self,xt):
        """
        """
        # Initialization of the output
        nt,C = xt.shape[0],len(self.prop)
        P = sp.empty((nt,C))

        for c in xrange(C):
        # Compute the posterior
            K = self.model[c].predict(xt,out='ki')
            K *= (-0.5)
            K[K>HDDA.E_MAX] = HDDA.E_MAX
            sp.exp(K,out=K)
            P[:,c]=self.prop[c]*K.sum(axis=1)
        
        # Get the class with the largest probability
        yp = sp.argmax(P,1)+1
        return yp
    
    def cross_validation(self,x,y,th,MODEL=['M1','M2','M3','M4','M5','M6','M7','M8'],C=[1,2,3,4,5,6],v=5,random_state=0):
        """
        """
        def get_kappa(confu):
            """
                Compute Kappa
            """
            n = sp.sum(confu)
            nl = sp.sum(confu,axis=1)
            nc = sp.sum(confu,axis=0)
            OA = sp.sum(sp.diag(confu))/float(n)

            return ((n**2)*OA - sp.sum(nc*nl))/(n**2-sp.sum(nc*nl))

        n,d = x.shape
        nth = len(th)
        
        ## Initialization of the stratified K-fold
        KF = StratifiedKFold(y.reshape(y.size,),v,random_state=random_state)
        Kappa = sp.zeros((nth,))

        for train,test in KF:
            for i,th_ in enumerate(th):# TODO: Change model to mda
                model = MDA()
                model.fit(x[train,:],y[train],MODEL=MODEL,th=[th_],C=C)
                yp = model.predict(x[test,:])
                confu = confusion_matrix(y[test],yp)
                Kappa[i] += get_kappa(confu)
                
        Kappa/=v
        ind = sp.argmax(Kappa)
        self.th  = [th[ind]]
        self.Kappa = Kappa
        # Learn model with optimal parameter
        self.fit(x,y,MODEL=MODEL,th=self.th,C=C)
