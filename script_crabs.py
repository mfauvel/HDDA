import hdda
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import PCA
from sklearn import mixture

# Parameters for HDDA
MODEL = 'M2'
C = 4  # For the example with do not fit the number of classes
th = 0.01  # The threshold for the Cattel test
tol = 0.00001
data = sp.load('crabs.npz')
X = data['x']
Y = data['y']

plt.figure()
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y, s=40)
plt.savefig('2D_true_labels.png')

bic, icl = [], []
for model_ in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']:
    model = hdda.HDDC(C=C, th=th, model=model_,
                      init='random', tol=tol)
    model.fit(X)
    bic.append(model.bic)
    icl.append(model.icl)

plt.figure()
plt.plot(bic)
plt.plot(icl)
plt.legend(("BIC", "ICL"))
plt.xticks(sp.arange(8), ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'))
plt.grid()
plt.savefig('bic_icl_crabs.png')

model = hdda.HDDC(C=C, th=th, model=MODEL, tol=tol)
model.fit(X)
model.bic
yp = model.predict(X)

plt.figure()
plt.scatter(Xp[:, 0], Xp[:, 1], c=yp, s=40)
plt.savefig("2D_hdda.png")

clf = mixture.GaussianMixture(n_components=4, covariance_type='full')
clf.fit(X)
yp = clf.predict(X)

plt.figure()
plt.scatter(Xp[:, 0], Xp[:, 1], c=yp, s=40)
plt.savefig('2D_gmm.png')
