import hdda
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import PCA
from sklearn import mixture

# Parameters for HDDA
MODEL = ['M1','M2','M3','M4','M5','M6','M7','M8']
th = sp.linspace(0.01,0.3,num=5)
param ={'init':'kmeans','C':4}

data = sp.load('crabs.npz')
X = data['x']
Y = data['y']

plt.figure()
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)
plt.scatter(Xp[:,0],Xp[:,1],c=Y,s=40)
plt.savefig('2D_true_labels.png')

BIC,ICL = [],[]
for MODEL in ('M1','M2','M3','M4','M5','M6','M7','M8'):
    print MODEL
    bic= sp.empty((th.size,1))
    icl= sp.empty((th.size,1))
    for i,th_ in enumerate(th):
        param['th']=th_
        model = hdda.HDGMM(model=MODEL)
        model.fit(X,param=param)
        bic[i],icl[i]=model.bic,model.icl
        
    t = sp.where(bic==bic.min())[0]
    BIC.append([bic.min(),MODEL,th[t[0]]])
    t = sp.where(icl==icl.min())[0]
    ICL.append([icl.min(),MODEL,th[t[0]]])

for bic,icl in zip(BIC,ICL):
    print bic
    print icl



# t = sp.argmin(BIC)
# best_model = MODEL[t]
# param = {'th':th[POS[t]],'C':4,'init':'kmeans'}
# model=hdda.HDGMM(model=best_model)
# yp=model.fit(X,param=param)

# print "Best model "+best_model
# print "With parameter " + str(th[POS[t]])

# plt.figure()
# plt.scatter(Xp[:,0],Xp[:,1],c=yp,s=40)
# plt.savefig("2D_hdda.png")

# clf = mixture.GMM(n_components=4, covariance_type='full')
# clf.fit(X)
# yp=clf.predict(X)

# plt.figure()
# plt.scatter(Xp[:,0],Xp[:,1],c=yp,s=40)
# plt.savefig('2D_gmm.png')
