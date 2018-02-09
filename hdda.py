# -*- coding: utf-8 -*-
import scipy as sp
from scipy import linalg
from sklearn.cluster import KMeans
from scipy.linalg.blas import dsyrk
from sklearn.utils.validation import check_array
from mpi4py import MPI
# TODO: Define get_param and set_param function
# TODO: Check the projection in predict -> could be faster ...
# TODO: add predict_proba function

# Numerical precision - Some constant
EPS = sp.finfo(sp.float64).eps
MIN = sp.finfo(sp.float64).min

# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()


# HDDC
class HDDC():
    """
    This class implements the HDDA models proposed by Charles Bouveyron
    and Stephane Girard
    Details about methods can be found here:
    https://doi.org/10.1016/j.csda.2007.02.009
    """

    def __init__(self, comm, rank,
                 model='M1', th=0.1, init='random',
                 itermax=100, tol=0.001, C=4,
                 population=None, random_state=0,
                 check_empty=False):
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
        self.n = None
        self.d = None
        self.th = th
        self.init = init
        self.itermax = itermax
        self.tol = tol
        self.C = C
        self.population = population
        self.random_state = random_state
        self.check_empty = check_empty  # Check for empty classes
        self.C_ = [C]  # List of clusters number w.r.t iterations
        self.Xm = None  # Sample mean

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
        self.aic = []         # aic values of the model
        self.icl = []         # icl values of the model
        self.niter = None     # Number of iterations
        self.dL = []          # Common covariance matrix eigenvalues
        self.T = []           # Membership matrix

        self.comm = comm
        self.rank = rank

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

        # Initialization
        np, d = X.shape
        np = sp.array(np)
        n = sp.empty_like(np)
        self.comm.Allreduce(np, n, op=MPI.SUM)
        self.n = n
        self.d = d
        LL = []
        ITER = 0

        X = check_array(X, copy=False, order='C', dtype=sp.float64)

        # Compute constant
        self.cst = self.d*sp.log(2*sp.pi)

        # Set minimum clusters size
        # Rule of dumbs for minimal size pi = 1 :
        # one mean vector (d) + one eigenvalues/vectors (1 + d)
        # + noise term (1) ~ 2(d+1)
        if self.population is None:
            self.population = 2*(self.d+1)

        if self.population > self.n/self.C:
            print("Number of classes to high w.r.t the number of samples:"
                  "C should be deacreased")
            return - 2

        # Initialization of the clustering
        if self.C == 1:
            self.T = sp.ones((np, 1))
        else:
            if self.init == 'kmeans':
                label = KMeans(n_clusters=self.C,
                               n_init=1, n_jobs=-1,
                               random_state=self.random_state).fit(X).labels_
                label += 1  # Label starts at one
            elif self.init == 'random':
                sp.random.seed(self.random_state)
                label = sp.random.randint(1, high=self.C+1, size=np)
            elif self.init == 'user':
                if self.C != y.max():
                    print("The number of class does not"
                          "match between self.C and y")
                label = y
            else:
                print("Initialization should be kmeans or random or user")
                return - 2  # Bad init values

            # Convert label to membership
            self.T = sp.zeros((np, self.C))
            self.T[sp.arange(np), label-1] = 1

        # Compute the whole covariance matrix and its eigenvalues if needed
        if self.model in ('M2', 'M4', 'M6', 'M8'):
            Xmp = sp.sum(X, axis=0)
            Xm = sp.empty_like(Xmp)
            self.comm.Allreduce(Xmp, Xm, op=MPI.SUM)  # Collective summation
            Xm /= self.n  # Global mean
            X_ = (X - Xm)
            # Use dsyrk to take benefit of the product symmetric matrices
            # X^{t}X or XX^{t}
            # Transpose to put in fortran order
            Wp = dsyrk(1.0/self.n, X_.T, trans=False)
            W = sp.empty_like(Wp)
            self.comm.Allreduce(Wp, W, op=MPI.SUM)
            del Wp, Xmp
            if self.rank == 0:
                # Compute intrinsic dimension on the whole data set
                L = linalg.eigh(W, eigvals_only=True, lower=False)
                idx = L.argsort()[::-1]
                L = L[idx]
                # Chek for numerical errors
                L[L < EPS] = EPS
                self.dL = sp.absolute(sp.diff(L))
                self.dL /= self.dL.max()
                del W, L
            else:
                self.dL = sp.empty((self.d))
            self.comm.Bcast(self.dL, root=0)

        # Initialization of the parameter
        self.m_step(X)
        llp = sp.array(self.e_step(X))
        ll = sp.empty_like(llp)
        self.comm.Allreduce(llp, ll, op=MPI.SUM)
        LL.append(float(ll))

        # Main while loop
        while(ITER < self.itermax):
            # M step
            self.free()
            self.m_step(X)

            # E step
            llp = sp.array(self.e_step(X))
            ll = sp.empty_like(llp)
            self.comm.Allreduce(llp, ll, op=MPI.SUM)
            LL.append(float(ll))

            if (abs((LL[-1]-LL[-2])/LL[-2]) < self.tol) and \
               (self.C_[-2] == self.C_[-1]):
                break
            else:
                ITER += 1

        # Return the class membership and some parameters of the optimization
        self.LL = LL
        self.bic = - 2*LL[-1] + self.q*sp.log(self.n)
        self.aic = - 2*LL[-1] + 2*self.q
        # Add small constant to ICL to prevent numerical issues
        self.icl = self.bic - 2*sp.log(self.T.max(axis=1)+EPS).sum()
        self.niter = ITER + 1

        # Remove temporary variables
        self.T = None
        return self

    def m_step(self, X):
        """M step of the algorithm

        This function  computes the  empirical estimators of  the mean
        vector,  the convariance  matrix  and the  proportion of  each
        class.

        """
        # Learn the model for each class
        C_ = self.C
        c_delete = []
        for c in xrange(self.C):
            nip = sp.array(self.T[:, c].sum())
            ni = sp.empty_like(nip)
            self.comm.Allreduce(nip, ni, op=MPI.SUM)  # Global reduction for ni
            # Check if empty
            if self.check_empty and \
               ni < self.population:
                C_ -= 1
                c_delete.append(c)
            else:
                self.ni.append(ni)
                self.prop.append(float(self.ni[-1])/self.n)
                Xmp = sp.dot(self.T[:, c].T, X)/self.ni[-1]
                Xm = sp.empty_like(Xmp)
                self.comm.Allreduce(Xmp, Xm, op=MPI.SUM)
                self.mean.append(Xm)
                X_ = (X-self.mean[-1])*(sp.sqrt(self.T[:, c])[:, sp.newaxis])

                # Use dsyrk to take benefit of symmetric matrices
                covp = dsyrk(1.0/(self.ni[-1]-1), X_.T, trans=False)
                cov = sp.empty_like(covp)
                self.comm.Allreduce(covp, cov, op=MPI.SUM)

                if self.rank == 0:
                    # Only the upper part of cov is initialize -> dsyrk
                    L, Q = linalg.eigh(cov, lower=False)
                else:
                    L = sp.empty((self.d))
                    Q = sp.empty((self.d, self.d))
                self.comm.Bcast(L, root=0)
                self.comm.Bcast(Q, root=0)

                # Chek for numerical errors
                L[L < EPS] = EPS
                if self.check_empty and (L.max() - L.min()) < EPS:
                    # In that case all eigenvalues are equal
                    # and this does not match the model
                    C_ -= 1
                    c_delete.append(c)
                    del self.ni[-1]
                    del self.prop[-1]
                    del self.mean[-1]
                else:
                    idx = L.argsort()[::-1]
                    L, Q = L[idx], Q[:, idx]

                    self.L.append(L)
                    self.Q.append(Q)
                    self.trace.append(cov.trace())

        # Update T
        if c_delete:
            self.T = sp.delete(self.T, c_delete, axis=1)

        # Update the number of clusters
        self.C_.append(C_)
        self.C = C_

        if self.rank == 0:
            # Estimation of the signal subspace
            # Common size subspace models for specific size subspace models
            if self.model in ('M1', 'M3', 'M5', 'M7'):
                for c in xrange(self.C):
                    # Scree test
                    dL, pi = sp.absolute(sp.diff(self.L[c])), 1
                    dL /= dL.max()
                    while sp.any(dL[pi:] > self.th):
                        pi += 1
                    if (pi < (min(self.ni[c], self.d) - 1)) and (pi > 0):
                        self.pi.append(pi)
                    else:
                        self.pi.append(1)
            elif self.model in ('M2', 'M4', 'M6', 'M8'):
                dL, p = self.dL, 1
                while sp.any(dL[p:] > self.th):
                    p += 1
                min_dim = int(min(min(self.ni), self.d))
                # Check if (p >= ni-1 or d-1) and p > 0
                if p < (min_dim - 1):
                    self.pi = [p for c in xrange(self.C)]
                else:
                    self.pi = [max((min_dim-2), 1) for c in xrange(self.C)]
                del dL, p, idx

            # Estim signal part
            self.a = [sL[:sPI] for sL, sPI in zip(self.L, self.pi)]
            if self.model in ('M5', 'M6', 'M7', 'M8'):
                self.a = [sp.repeat(sA[:].mean(), sA.size) for sA in self.a]

            # Estim noise term
            if self.model in ('M1', 'M2', 'M5', 'M6'):
                # Noise free
                self.b = [(sT-sA.sum())/(self.d-sPI)
                          for sT, sA, sPI in zip(self.trace, self.a, self.pi)]
                # Check for very small value of b
                self.b = [b if b > EPS else EPS for b in self.b]

            elif self.model in ('M3', 'M4', 'M7', 'M8'):
                # Noise common
                denom = self.d - sp.sum([sPR*sPI
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
            self.logdet = [(sp.log(sA).sum() + (self.d-sPI)*sp.log(sB))
                           for sA, sPI, sB in
                           zip(self.a, self.pi, self.b)]

            # Update the Q matrices
            self.Q = [sQ[:, :sPI]
                      for sQ, sPI in
                      zip(self.Q, self.pi)]

            # Compute the number of parameters of the model
            self.q = self.C*self.d + (self.C-1) + sum([sPI*(self.d-(sPI+1)/2)
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

        # Broadcast value
        self.pi = self.comm.bcast(self.pi, root=0)
        self.a = self.comm.bcast(self.a, root=0)
        self.b = self.comm.bcast(self.b, root=0)
        self.logdet = self.comm.bcast(self.logdet, root=0)
        self.Q = self.comm.bcast(self.Q, root=0)
        self.q = self.comm.bcast(self.q, root=0)

    def e_step(self, X):
        """Compute the e-step of the algorithm

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------

        """
        # Get some parameters
        n = X.shape[0]

        # Compute the membership function
        K = self.score_samples(X)

        # Compute the Loglikelhood
        K *= (0.5)
        Km = K.max(axis=1)
        Km.shape = (n, 1)

        # logsumexp trick
        LL = (sp.log(sp.exp(K-Km).sum(axis=1))[:, sp.newaxis]+Km).sum()

        # Compute the posterior
        with sp.errstate(over='ignore'):
            for c in xrange(self.C):
                self.T[:, c] = 1 / sp.exp(K-K[:, c][:, sp.newaxis]).sum(axis=1)

        return LL

    def score(self, X, y=None):
        """Compute the per-sample log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        log_likelihood : float
            Log likelihood of the Gaussian mixture given X.

        """

        X = check_array(X, copy=False, order='C', dtype=sp.float64)

        # Get some parameters
        n = X.shape[0]

        # Compute the membership function
        K = self.score_samples(X)

        # Compute the Loglikelhood
        K *= (0.5)
        Km = K.max(axis=1)
        Km.shape = (n, 1)
        # logsumexp trick
        LL = (sp.log(sp.exp(K-Km).sum(axis=1))[:, sp.newaxis]+Km).sum()

        return LL

    def score_samples(self, X, y=None):
        """Compute the negative weighted log probabilities for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples, n_clusters)
            Log probabilities of each data point in X.
        """
        X = check_array(X, copy=False, order='C', dtype=sp.float64)
        nt, d = X.shape
        K = sp.empty((nt, self.C))
        # Start the prediction for each class
        for c in xrange(self.C):
            # Compute the constant term
            K[:, c] = self.logdet[c] - 2*sp.log(self.prop[c]) + self.cst

            # Remove the mean
            Xc = X - self.mean[c]

            # Do the projection
            Px = sp.dot(Xc,
                        sp.dot(self.Q[c], self.Q[c].T))
            temp = sp.dot(Px, self.Q[c]/sp.sqrt(self.a[c]))
            K[:, c] += sp.sum(temp**2, axis=1)
            K[:, c] += sp.sum((Xc - Px)**2, axis=1)/self.b[c]

        return -K

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = check_array(X, copy=False, order='C', dtype=sp.float64)
        return self.score_samples(X).argmax(axis=1) + 1

    def free(self):
        """This  function free some  parameters of the  model.

        Use in the EM algorithm
        """
        self.pi = []
        self.a = []
        self.b = []
        self.logdet = []
        self.q = []

        self.ni = []          # Number of samples of each class
        self.prop = []        # Proportion of each class
        self.mean = []        # Mean vector
        self.pi = []            # Signal subspace size
        self.L = []           # Eigenvalues of covariance matrices
        self.Q = []
        self.trace = []
