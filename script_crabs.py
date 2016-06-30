import hdda
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import PCA
from sklearn import mixture

# Parameters for HDDA
MODEL = ['M1','M2','M3','M4','M5','M6','M7','M8']
C = [4] # For the example with do not fit the number of classes
th = [0.05,0.1,0.2] # The threshold for the Cattel test

data = sp.load('crabs.npz')
X = data['x']
Y = data['y']

plt.figure()
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)
plt.scatter(Xp[:,0],Xp[:,1],c=Y,s=40)
plt.savefig('2D_true_labels.png')

model = hdda.HDGMM()
model.fit_all(X,MODEL=MODEL,C=C,th=th,VERBOSE=True)

yp=model.predict(X)

plt.figure()
plt.scatter(Xp[:,0],Xp[:,1],c=yp,s=40)
plt.savefig("2D_hdda.png")

clf = mixture.GMM(n_components=4, covariance_type='full')
clf.fit(X)
yp=clf.predict(X)

plt.figure()
plt.scatter(Xp[:,0],Xp[:,1],c=yp,s=40)
plt.savefig('2D_gmm.png')
