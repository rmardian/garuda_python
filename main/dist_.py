import matplotlib
import matplotlib.pyplot as plt
from numpy.random import normal

healthy_mean = 0.9
healthy_sd   = 0.3

sick_mean    = 0.4
sick_sd      = 0.4 

healthy_growthrates = normal(size=50,loc=healthy_mean,scale=healthy_sd)
sick_growthrates = normal(size=50,loc=sick_mean,scale=sick_sd)

plt.hist(healthy_growthrates, bins=20, histtype='stepfilled', normed=True, color='b', label='Healthy')
plt.hist(sick_growthrates, bins=20, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Sick')
plt.title("Healthy vs. sick")
plt.xlabel("Growth Rate")
plt.ylabel("Probability")
plt.legend()
plt.show()