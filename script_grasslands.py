for b in {B,G,R,NIR}
do
    echo [[file:grassland_$b.png]]
done

import hdda
import scipy as sp
import matplotlib.pyplot as plt
import time as time

# Load data
data = sp.load('prairie5.npy')
x = data
n,d=x.shape
print "Number of samples: {}\n Number of variables: {}".format(n,d)
# Parameters
MODEL = ['M1','M2','M3','M4','M5','M6','M7','M8']
th = [0.05,0.1,0.2]
C = sp.arange(1,5)

# Model Selection
model = hdda.HDGMM()
tic = time.clock()
model.fit_all(x,MODEL=MODEL,C=C,th=th,VERBOSE=True)
toc = time.clock()
print "Processing time: {}".format(toc-tic)

# Plot data
bands= ['B','G','R','NIR']

for i,b in enumerate(bands):
    plt.figure()
    # Plot the samples
    for j in xrange(n):
        plt.plot(data[j,(i*17):((i+1)*17)],'k',lw=0.5)
    # Plot the means
    for j in xrange(len(model.mean)):
        plt.plot(model.mean[j][(i*17):((i+1)*17)],lw=3)
    plt.savefig('grassland_{}.png'.format(b))
