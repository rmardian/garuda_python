import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import beta

import matplotlib.mlab as mlab
import numpy as np
from numpy.random import normal
from scipy.optimize import curve_fit
import csv

#read-data
with open('label-complete-424.csv') as csvfile:
    y = list(csv.reader(csvfile))
label_temp = [[float(j) for j in i] for i in y]
label = [val for sublist in label_temp for val in sublist]

mu = np.mean(label)
sig = np.std(label)

#gaussian-function
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

n, bins, patches = plt.hist(label, bins=20, normed=True)

xt = plt.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(label))

# lets try the normal distribution first
m, s = norm.fit(label) # get mean and standard deviation  
pdf_g = norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
plt.plot(lnspc, pdf_g, label="Norm") # plot it

#ag,bg,cg = gamma.fit(label)
#pdf_gamma = gamma.pdf(lnspc, ag, bg,cg)
#plt.plot(lnspc, pdf_gamma, label="Gamma")

#ab,bb,cb,db = beta.fit(label)
#pdf_beta = beta.pdf(lnspc, ab, bb,cb, db)
#plt.plot(lnspc, pdf_beta, label="Beta")

#y = mlab.normpdf(bins, mu, sig)
#l = plt.plot(bins, y, 'r--', linewidth=2)

plt.show()