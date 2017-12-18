# -*- coding: utf-8 -*-
import scipy as sp
from scipy import linalg
from sklearn.cluster import KMeans
from scipy.linalg.blas import dsyrk


# TODO: clean the output of predict when out=proba,  add the posterior probabilities
# TODO: Work on ni rather than n for selected the number of eigenvalues -> needs to re-define check for the values of pi
# TODO: Work on return values for checking errors
# TODO: Add check_array function for fit and predict function (in particular score=bic)
# TODO: Change name of functions to match "APIs of scikit-learn objects"
# TODO: Define get_param and set_param function
# TODO: Empty classes

# Numerical precision - Some constant
EPS = sp.finfo(sp.float64).eps
MIN = sp.finfo(sp.float64).min


# HDDC
class HDDC():
    """
    This class implements the HDDA models proposed by Charles Bouveyron
    and Stephane Girard
    Details about methods can be found here:
    https://doi.org/10.1016/j.csda.2007.02.009
    """

    def __init__(self, model='M1', th=0.1, init='kmeans',
                 itermax=100, tol=0.001, C=4,
                 population=2, random_state=None):
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
        # Hyperparameters of the algorithm
        self.th = th
        self.init = init
        self.itermax = itermax
        self.tol = tol
        self.C = C
        self.population = population
        self.random_state = random_state

        self.ni = []  # Number of samples of each class
        self.prop = []  # Proportion of each class
        self.mean = []  # Mean vector
        self.pi = []  # Signal subspace size
        self.L = []  # Eigenvalues of covariance matrices
        self.Q = []  # Eigenvectors of covariance matrices
        self.trace = []  # Trace of the covariance matrices
        self.a = []  # Eigenvalues of signal subspaces
        self.b = []  # Values of the noise
        self.logdet = []  # Pre-computation of the logdet
        if model in ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'):
            self.model = model  # Name of the model
        else:
            print "Model parameter {} is not available".format(model)
            exit()
        self.q = []           # Number of parameters of the full models
        self.bic = []         # bic values of the model
        self.icl = []         # icl values of the model
        self.niter = None     # Number of iterations
        self.X = []           # Matrix to project samples when n<d
        self.W = []           # Common covariance matrix
        self.T = []           # Membership matrix

    def fit(self, X, y=None):
        """Estimate the model parameters with the EM algorithm

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        self
        """
        n, d = X.shape
        LL = []
        ITER = 0
        if self.C == 1:
            self.T = sp.ones((n, 1))
        else:
            if self.init == 'kmeans':
                label = KMeans(n_clusters=self.C,
                               n_init=1, n_jobs=-1,
                               random_state=self.random_state).fit(X).labels_
                # Check for minimal size of cluster
                nc = sp.asarray([len(sp.where(label == i)[0])
                                 for i in xrange(self.C)])
                if sp.any(nc < 2):
                    self.LL, self.bic, self.icl, self.niter \
                        = LL, MIN, MIN, (ITER+1)
                    return -1  # Kmeans failed
                else:
                    label += 1  # Label starts at one
            elif self.init == 'random':
                sp.random.seed(self.random_state)
                label = sp.random.randint(1, high=self.C+1, size=n)
            elif self.init == 'user':
                if self.C != y.max():
                    print("The number of class does not match between self.C and y")                
                label = y
            else:
                print("Initialization should be kmeans or random or user")
                return - 2  # Bad init values

            # Convert label to membership
            self.T = sp.zeros((n, self.C))
            self.T[sp.arange(n), label-1] = 1

        # Initialization of the parameter
        self.m_step(X)
        ll = self.loglike(X)
        LL.append(ll)
        while(ITER < self.itermax):
            # E step - Use the precomputed T

            # Check for empty classes
            if sp.any(self.T.sum(axis=0) < self.population):
                self.LL, self.bic, self.icl, self.niter\
                    = LL, MIN, MIN, (ITER+1)
                return - 3  # population empty

            # M step
            self.free(full=True)
            self.m_step(X)

            # Compute the BIC and do the E step - update T
            ll = self.loglike(X)
            LL.append(ll)
            if abs((LL[-1]-LL[-2])/LL[-2]) < self.tol:
                break
            else:
                ITER += 1

        # Return the class membership and some parameters of the optimization
        self.LL = LL
        self.bic = 2*LL[-1] - self.q*sp.log(n)
        # Add small constant to prevent numerical issues
        self.icl = self.bic + 2*sp.log(self.T.max(axis=1)+EPS).sum()
        self.niter = ITER + 1

        return self

    def m_step(self, X):
        """M step of the algorithm

        This function  computes the  empirical estimators of  the mean
        vector,  the convariance  matrix  and the  proportion of  each
        class.

        """

        # Get information from the data
        n, d = X.shape    # Number of samples and number of variables

        # Compute constant
        self.cst = d*sp.log(2*sp.pi)

        # Compute the whole covariance matrix
        if self.model in ('M2', 'M4', 'M6', 'M8'):
            X_ = (X - sp.mean(X, axis=0))
            # Use dsyrk to take benefit of the product symmetric matrices
            # X^{t}X or XX^{t}
            # Transpose to put in fortran order
            if n >= d:
                self.W = dsyrk(1.0/n, X_.T, trans=False)
            else:
                self.W = dsyrk(1.0/n, X_.T, trans=True)
            X_ = None

        # Learn the model for each class
        for c in xrange(self.C):
            self.ni.append(self.T[:, c].sum())
            self.prop.append(float(self.ni[c])/n)
            self.mean.append(sp.dot(self.T[:, c].T, X)/self.ni[c])
            X_ = (X-self.mean[c])*(sp.sqrt(self.T[:, c])[:, sp.newaxis])

            # Use dsyrk to take benefit of the product of symmetric matrices
            if n >= d:
                cov = dsyrk(1.0/(self.ni[c]-1), X_.T, trans=False)
            else:
                cov = dsyrk(1.0/(self.ni[c]-1), X_.T, trans=True)
                self.X.append(X_)
            X_ = None

            # Only the upper part of cov is initialize -> dsyrk
            L, Q = linalg.eigh(cov, lower=False)
            idx = L.argsort()[::-1]
            L, Q = L[idx], Q[:, idx]
            # Chek for numerical errors
            L[L < EPS] = EPS

            self.L.append(L)
            self.Q.append(Q)
            self.trace.append(cov.trace())

        # Estimation of the signal subspace
        # Common size subspace models
        if self.model in ('M2', 'M4', 'M6', 'M8'):
            # Compute intrinsic dimension on the whole data set
            L = linalg.eigh(self.W, eigvals_only=True, lower=False)
            idx = L.argsort()[::-1]
            L = L[idx]
            # Chek for numerical errors
            L[L < EPS] = EPS
            # To take into account python broadcasting a[:p] = a[0]...a[p-1]
            dL, p = sp.absolute(sp.diff(L)), 1
            dL /= dL.max()
            while sp.any(dL[p:] > self.th):
                p += 1
            min_dim = int(min(min(self.ni), d))
            # Check if (p >= ni-1 or d-1) and p > 0
            if p < (min_dim - 1):
                self.pi = [p for c in xrange(self.C)]
            else:
                self.pi = [max((min_dim-2), 1) for c in xrange(self.C)]

        # Specific size subspace models
        elif self.model in ('M1', 'M3', 'M5', 'M7'):
            for c in xrange(self.C):
                # Scree test
                dL, pi = sp.absolute(sp.diff(self.L[c])), 1
                dL /= dL.max()
                while sp.any(dL[pi:] > self.th):
                    pi += 1
                if (pi < (min(self.ni[c], d) - 1)) and (pi > 0):
                    self.pi.append(pi)
                else:
                    self.pi.append(1)

        # Estim signal part
        self.a = [sL[:sPI] for sL, sPI in zip(self.L, self.pi)]
        if self.model in ('M5', 'M6', 'M7', 'M8'):
            self.a = [sp.repeat(sA[:].mean(), sA.size) for sA in self.a]

        # Estim noise term
        if self.model in ('M1', 'M2', 'M5', 'M6'):
            # Noise free
            self.b = [(sT-sA.sum())/(d-sPI)
                      for sT, sA, sPI in zip(self.trace, self.a, self.pi)]
            # Check for very small value of b
            self.b = [b if b > EPS else EPS for b in self.b]

        elif self.model in ('M3', 'M4', 'M7', 'M8'):
            # Noise common
            denom = d - sp.sum([sPR*sPI
                                for sPR, sPI in
                                zip(self.prop, self.pi)])
            num = sp.sum([sPR*(sT-sA.sum())
                          for sPR, sT, sA in
                          zip(self.prop, self.trace, self.a)])

            # Check for very small values
            if num < EPS:
                self.b = [EPS for i in xrange(self.C)]
            elif denom < EPS:
                self.b = [1/EPS for i in xrange(self.C)]
            else:
                self.b = [num/denom for i in xrange(self.C)]

        # Compute remainings parameters
        # Precompute logdet
        self.logdet = [(sp.log(sA).sum() + (d-sPI)*sp.log(sB))
                       for sA, sPI, sB in
                       zip(self.a, self.pi, self.b)]

        # Update the Q matrices
        if n >= d:
            self.Q = [sQ[:, :sPI]
                      for sQ, sPI in
                      zip(self.Q, self.pi)]
        else:
            self.Q = [sp.dot(sX.T, sQ[:, :sPI])/sp.sqrt(sL[:sPI])
                      for sX, sQ, sPI, sL in
                      zip(self.X, self.Q, self.pi, self.L)]

        # Compute the number of parameters of the model
        self.q = self.C*d + (self.C-1) + sum([sPI*(d-(sPI+1)/2)
                                              for sPI in self.pi])
        # Number of noise subspaces
        if self.model in ('M1', 'M3', 'M5', 'M7'):
            self.q += self.C
        elif self.model in ('M2', 'M4', 'M6', 'M8'):
            self.q += 1
        # Size of signal subspaces
        if self.model in ('M1', 'M2'):
            self.q += sum(self.pi) + self.C
        elif self.model in ('M3', 'M4'):
            self.q += sum(self.pi) + 1
        elif self.model in ('M5', 'M6'):
            self.q += 2*self.C
        elif self.model in ('M7', 'M8'):
            self.q += self.C+1

    def loglike(self, X):
        """
        Compute the log likelyhood given a set of samples.
        Update the belongship vector
        :param X: The sample matrix,
        """

        # Get some parameters
        n = X.shape[0]

        # Compute the membership function
        K = self.score_samples(X)

        # Compute the Loglikelhood
        K *= (-0.5)
        Km = K.max(axis=1)
        Km.shape = (n, 1)
        # logsumexp trick
        LL = (sp.log(sp.exp(K-Km).sum(axis=1))[:, sp.newaxis]+Km).sum()

        # Compute the posterior
        with sp.errstate(over='ignore'):
            for c in xrange(self.C):
                self.T[:, c] = 1 / sp.exp(K-K[:, c][:, sp.newaxis]).sum(axis=1)

        return LL

    def score_samples(self, Xt):
        """
        This function compute the decision of the fitted HD model.
        :param Xt: The samples matrix of testing samples
        :type xt: float
        :return K: Log probabilities of each data point in Xt
        """
        nt, d = Xt.shape
        K = sp.empty((nt, self.C))

        # Start the prediction for each class
        for c in xrange(self.C):
            # Compute the constant term
            K[:, c] = self.logdet[c] - 2*sp.log(self.prop[c]) + self.cst

            # Remove the mean
            Xtc = Xt - self.mean[c]

            # Do the projection
            Px = sp.dot(Xtc,
                        sp.dot(self.Q[c], self.Q[c].T))
            temp = sp.dot(Px, self.Q[c]/sp.sqrt(self.a[c]))
            K[:, c] += sp.sum(temp**2, axis=1)
            K[:, c] += sp.sum((Xtc - Px)**2, axis=1)/self.b[c]

        return K

    def predict(self, Xt):
        """Predict the labels for the data samples in X using trained model.
        """
        return self.score_samples(Xt).argmin(axis=1) + 1

    def free(self, full=False):
        """This  function free some  parameters of the  model. It is  used to
        speed-up the cross validation process.

        :param full: To free only the parcimonious part or all the model
        :type full: bool
        """
        self.pi = []
        self.a = []
        self.b = []
        self.logdet = []
        self.q = []

        if full:
            self.ni = []          # Number of samples of each class
            self.prop = []        # Proportion of each class
            self.mean = []        # Mean vector
            self.pi = []            # Signal subspace size
            self.L = []           # Eigenvalues of covariance matrices
            self.Q = []
            self.trace = []
            self.X = []
            self.W = None
