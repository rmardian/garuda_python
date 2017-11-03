import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal

lines = [line.rstrip('\n') for line in open('healthy.txt')]

x = np.array(lines, dtype='|S4')
h = x.astype(np.float)

lines = [line.rstrip('\n') for line in open('toxic.txt')]

x = np.array(lines, dtype='|S4')
t = x.astype(np.float)

#print y

healthy_mean = 1.0
healthy_sd   = 0.1

sick_mean    = 0.0
sick_sd      = 0.1

healthy_growthrates = normal(size=1000,loc=healthy_mean,scale=healthy_sd)

sick_growthrates = normal(size=1000,loc=sick_mean,scale=sick_sd)

cummulative = np.concatenate([h, t])

plt.figure(1)

plt.hist(t, bins=20, histtype='stepfilled', normed=True, color='r', label='Sick')
plt.hist(h, bins=20, histtype='stepfilled', normed=True, color='b', alpha=0.5, label='Healthy')

plt.title("Healthy vs. sick")
plt.xlabel("Growth Rate")
plt.ylabel("Probability")
plt.legend()

plt.figure(2)

plt.hist(cummulative, bins=20, histtype='stepfilled', normed=True, color='g', label='Cummulative (both)')

plt.title("Cummulative")
plt.xlabel("Growth Rate")
plt.ylabel("Probability")
plt.legend()


plt.show()