# -*- coding: utf-8 -*-
import scipy as sp
from scipy import linalg
from sklearn.cross_validation import StratifiedKFold
from sklearn.cluster import KMeans
import accuracy_index as ai

## Numerical precision
eps = sp.finfo(sp.float64).eps

## Empirical estimators for EM
def soft_cov(x,m,w):
    """
    TBC
    """
    n,d=x.shape
    w_ = sp.copy(w).reshape(n,1)
    # Center the data
    xc = x-m
    # Compute the normalization and check for small values
    w_sum=w.sum()
    if w_sum < eps:
        w_sum = eps
    # Compute the soft covariance matrix using <X.T,X>
    return sp.dot(xc.T,xc*w_)/w_sum

#----------------------------TODO-------------------------------------#
# TODO: Rajouter les quatres derniers modèles
# TODO: Add the other submodels dans la fonction CV
# TODO: Regarder une version "safe" pour le calcul des probabilités dans EM
#---------------------------------------------------------------------#

## HDDA Class
class HDGMM():
    """
    This implements the HDDA models proposed by Charles Bouveyron and Stephane Girard
    Details about methods can be found here:
    http://w3.mi.parisdescartes.fr/~cbouveyr/
    """
    def __init__(self,model='M1'): 
        """
        This function initialize the HDDA stucture
        :param model: the model used.
        :type mode: string
        - M1 = aijbiQidi
        - M2 = aijbiQid
        - M3 = aijbQidi
        - M4 = aijbQid
        - M5 = aibiQidi
        - M6 = aibiQid
        - M7 = aibQidi
        - M8 = aibQid
        - M9 = abiQidi <--
        - M10 = abiQid
        - M11 = abQidi
        - m12 = abQid
        """
        self.ni = []          # Number of samples of each class
        self.prop = []        # Proportion of each class
        self.mean = []        # Mean vector
        self.pi=[]            # Signal subspace size
        self.L = []           # Eigenvalues of covariance matrices
        self.Q = []           # Eigenvectors of covariance matrices
        self.a = []           # Eigenvalues of signal subspaces
        self.b = []           # Values of the noise
        self.logdet = []      # Pre-computation of the logdet of covariance matrices using HDDA models
        self.icov =[]         # Pre-computation of the inverse of covariance matrices using HDDA models
        self.model=model      # Name of the model
        self.q = []           # Number of parameters of the full models
        self.bic = []

    def free(self,full=None):
        """This  function free some  parameters of the  model. It is  used to
        speed-up the cross validation process.
        
        :param full: To free only the parcimonious part or all the model
        :type full: int
        """
        self.pi=[]
        self.a = []
        self.b = []
        self.logdet = []
        self.icov =[]
        self.q = []
        self.bic = []
        
        if full is not None:
            self.ni = []          # Number of samples of each class
            self.prop = []        # Proportion of each class
            self.mean = []        # Mean vector
            self.pi=[]            # Signal subspace size
            self.L = []           # Eigenvalues of covariance matrices
            self.Q = [] 
        
    def fit(self,x,y=None,param=None):
        """
        This function fit the HDDA model

        :param x: The sample matrix, is of size x \times d where n is the number of samples and d is the number of variables
        :param y: The vector of corresponding labels, is of size n \times 1 in the supervised case, otherwise it is None
        :param param: A dictionnary of parameters. For the supervised case, it contains the threshold or the size of the signal subspace. For the unsupervised case, it contains also the number of classes and the initialization method.
        :type x: float
        :type y: int
        :type param: dictionnary
        :return: the predicted label for the unsupervised case
        """
        EM = False
        n,d = x.shape

        # Set defaults parameters
        default_param={'th':0.9,'p':5,'init':'kmeans','itermax':100,'tol':0.0001,'C':4,'population':2}
        for key,value in default_param.iteritems():
            if not param.has_key(key):
                param[key]=value
                
        # If unsupervised case
        if y is None: # Initialisation of the class membership
            init = param['init']
            EM,ITER,ITERMAX,TOL = True,0,param['itermax'],param['tol']
            if init is 'kmeans':
                y = KMeans(n_clusters=param['C'],n_init=20,n_jobs=-2,random_state=0).fit_predict(x)
                y += 1 # Label starts at one
            elif init is 'random':
                y = sp.random.randint(1,high=param['C']+1,size=n)
            
        # Initialization of the parameter
        self.fit_init(x,y)
        self.fit_update(param)
        BIC_o = self.BIC(x,y)

        if EM is True: # Unsupervised case, needs iteration
            while(ITER<ITERMAX):
                # E step
                T = sp.exp(-0.5*self.predict(x,out='ki'))
                T /= sp.sum(T,axis=1).reshape(n,1)
                T[T<eps]=0

                # M step
                self.free(full=1)
                self.fit_init(x,T)
                self.fit_update(param)

                # Check for empty classes
                if sp.any(sp.asarray(self.ni)<param['population']):
                    break
                
                # Compute the BIC
                BIC_n = self.BIC(x,T)
                if (BIC_o-BIC_n)/BIC_o < TOL:
                    break
                else:
                    BIC_o = BIC_n
                    ITER += 1
            # Return the class membership
            self.bic = BIC_n
            return sp.argmax(T,1)+1
                
    def predict(self,xt,out=None):
        """
        This function compute the decision of the fitted HD model.
        :param xt: The samples matrix of testing samples
        :param out: Setting to a value different from None will let the function returns the posterior probability for each class.
        :type xt: float
        :type out: string
        :return yp: The predicted labels and posterior probabilities if asked.
        """
        nt = xt.shape[0]
        C = len(self.a)
        K = sp.empty((nt,C))
        
        ## Start the prediction for each class
        for c in range(C):
            cst = self.logdet[c] - 2*sp.log(self.prop[c])
            xtc = xt-self.mean[c]
            temp = sp.dot(xtc,self.icov[c])
            K[:,c]=sp.sum(xtc*temp,axis=1)+cst
            
        ## Assign the label to the minimum value of K 
        if out is None:
            yp = sp.argmin(K,1)+1
            return yp
        elif out is 'proba':
            for c in range(C):
                K[:,c] += 2*sp.log(self.prop[c])
            K *= -0.5
            return yp,K
        elif out is 'ki':
            return K        

    def fit_init(self,x,y):
        """This  function computes  the  empirical  estimators of  the  mean
        vector,  the convariance  matrix  and the  proportion of  each
        class.
        :param x: The sample matrix, is of size x \times d where n is the number of samples and d is the number of variables
        :param y: The vector of corresponding labels, is of size n \times 1 in the supervised case and n \times C in the unsupervised case
        :type x: float
        :type y: int
        """
        ## Get information from the data
        n = x.shape[0]  # Number of samples
        if y.ndim == 1:  # Number of classes
            C = int(y.max(0))   
        else:
            C = y.shape[1]
        
        if n != y.shape[0]:
            print("size of x and y should match")
            exit()

        ## Learn the empirical of the model for each class
        for c in xrange(C):
            if y.ndim == 1: # Supervised case
                j = sp.where(y==(c+1))[0]
                self.ni.append(j.size)
                self.prop.append(float(self.ni[c])/n)
                self.mean.append(sp.mean(x[j,:],axis=0))
                cov = sp.cov(x[j,:],rowvar=0)
            else: # Unsupervised case
                self.ni.append(y[:,c].sum())
                self.prop.append(float(self.ni[c])/n)
                self.mean.append(sp.average(x,weights=y[:,c],axis=0))
                cov = soft_cov(x,self.mean[c],y[:,c])
                
            L,Q = linalg.eigh(cov) # Compute the spectral decomposition
            idx = L.argsort()[::-1]
            L,Q=L[idx],Q[:,idx]
            L[L<eps]=eps
            self.L.append(L)
            self.Q.append(Q)      

    def fit_update(self,param):
        """
        This function compute the parcimonious HDDA model from the empirical estimates obtained with fit_init
        """
        C = len(self.ni)
        d = self.mean[0].size

        # Get parameters 
        if self.model in ('M1','M3','M5','M7'): 
            th = param['th']
        elif self.model in ('M2','M4','M6','M8'):
            p = param['p']
        
        for c in xrange(C):
            # Estimation of the signal subspace
            if self.model in ('M1','M3','M5','M7'):
                pi = sp.where(sp.cumsum(self.L[c])/sp.sum(self.L[c])>th)[0][0]+1   
            elif self.model in ('M2','M4','M6','M8'):
                pi = p
            if pi >= d: # Check for consistency of size: it should be at most "d-1"
                pi = d-1
            self.pi.append(pi)
            
        if self.model in ('M1','M2','M5','M6'): # Noise free
            for c in xrange(C):
                # Estim signal part
                self.a.append(self.L[c][:self.pi[c]])
                if self.model in ('M5','M6'):
                    self.a[c][:] = self.a[c][:].mean()
                 # Estim noise part
                self.b.append((self.L[c].sum()-self.a[c].sum())/(d-self.pi[c]))
                # Check for very small value of b
                if self.b[c]<eps: 
                    self.b[c]=eps
                # Compute logdet
                self.logdet.append(sp.log(self.a[c]).sum() + (d-self.pi[c])*sp.log(self.b[c])) 
                # Compute the inverse of the covariances matrix
                temp = self.Q[c][:,:self.pi[c]]*(1/self.a[c]-1/self.b[c]).reshape(self.pi[c])
                self.icov.append(sp.dot(temp,self.Q[c][:,:self.pi[c]].T)+sp.eye(d)/self.b[c])
                temp = []
                
        elif self.model in ('M3','M4','M7','M8'):# Noise common
            # Estimation of b
            denom = d - sum(map(lambda prop,pi:prop*pi,self.prop,self.pi))
            num = sum(map(lambda prop,pi,L:prop*L[pi:].sum(),self.prop,self.pi,self.L))
            # Check for very small value of b
            if num<eps:
                self.b = eps
            elif denom<eps:
                self.b = 1.0/eps
            else:
                self.b = num/denom               
            
            for c in xrange(C):
                # Estim signal part
                self.a.append(self.L[c][:self.pi[c]])
                if self.model in ('M7','M8'):
                    self.a[c][:] = self.a[c][:].mean()
                # Compute logdet
                self.logdet.append(sp.log(self.a[c]).sum() + (d-self.pi[c])*sp.log(self.b)) 
                # Compute the inverse of the covariances matrix
                temp = self.Q[c][:,:self.pi[c]]*(1/self.a[c]-1/self.b).reshape(self.pi[c]) 
                self.icov.append(sp.dot(temp,self.Q[c][:,:self.pi[c]].T)+sp.eye(d)/self.b)
                temp = []
                
        # Compute the number of parameters of the model
        self.q = C*d + (C-1) + sum(map(lambda p:p*(d-(p+1)/2),self.pi)) # Mean vectors + proportion + eigenvectors
        if self.model in ('M1','M3','M5','M7'): # Number of noise subspaces
            self.q += C 
        elif self.model in ('M2','M4','M6','M8'):
            self.q += 1 
        if self.model in ('M1','M2'): # Size of signal subspaces
            self.q += sum(self.pi)+C 
        elif self.model in ('M3','M4'):
            self.q += sum(self.pi)+ 1
        elif self.model in ('M5','M6'):
            self.q += 2*C
        elif self.model in ('M7','M8'):
            self.q += C+1               
        
    def CV(self,x,y,param,v=5,seed=0):
        """
        This function computes the cross validation estimate of the Kappa coefficient of agreement given a set of parameters in the supervised case. 
        To speed up the processing, the empirical estimate (mean, proportion, eigendecomposition) is done only one for each fold.
        
        :param x: The sample matrix, is of size x \times d where n is the number of samples and d is the number of variables
        :param y: The vector of corresponding labels, is of size n \times 1 in the supervised case, otherwise it is None
        :param param: A dictionnary of parameters.
        :param v: the number of folds of the CV.
        :param seed: the initial state of the random generator.
        :return: the optimal value for the given model and the corresponding Kappa
        """
        # Initialization of the stratified K-Fold
        KF = StratifiedKFold(y.reshape(y.size,),v,random_state=seed)

        # Get parameters grid
        if self.model in ('M1','M3','M5','M7'): # TODO: Add other models
            param_grid = param['th']
        elif self.model in ('M2','M4','M6','M8'):
            param_grid = param['p']
            
        # Initialize the confusion matrix and the Kappa coefficient vector
        acc,Kappa = ai.CONFUSION_MATRIX(),sp.zeros((len(param_grid)))
        for train,test in KF:
            modelTemp = HDGMM(model=self.model)
            modelTemp.fit_init(x[train,:],y[train])
            for i,param_grid_ in enumerate(param_grid):
                # Fit model on train subests
                if modelTemp.model in ('M1','M3','M5','M7'):
                    param_= {'th':param_grid_}
                elif modelTemp.model in ('M2','M4','M6','M8'):
                    param_= {'p':param_grid_}
                modelTemp.fit_update(param_)
                # Predict on test subset
                yp = modelTemp.predict(x[test,:])
                acc.compute_confusion_matrix(yp,y[test])
                Kappa[i] += acc.Kappa
                modelTemp.free()
        Kappa /= v
        # Select the value with the highest Kappa value
        ind = sp.argmax(Kappa)
        return param_grid[ind],Kappa[ind]

    def BIC(self,x,T):
        """
        Compute the BIC given a set of samples and its membership.
        :param x: The sample matrix, is of size x \times d where n is the number of samples and d is the number of variables
        :param T: the membership matrix
        """
        N = sum(self.ni)
        K = self.predict(x,out='ki')
        if T.ndim == 1:
            C = int(T.max())
            S =  sp.zeros((T.size,C))
            for i in range(1,C+1):
                t=sp.where(T==i)[0]
                S[t,i-1]=1
            LL = K*S
        else:
            LL = K*T    
        return (LL.sum() + self.q*sp.log(N))/N
