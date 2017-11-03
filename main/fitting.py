import numpy as np
#import bigfloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

def logarithmic(x, p1, p2):
	return p1*np.log(x)+p2
 
def sigmoid(x, x0, k):
	y = 1 / (1 + np.exp(-k*(x-x0)))
	#y = 1 / (1 + bigfloat.exp(-k*(x-x0),bigfloat.precision(100)))
	return y
      
#def sigmoid_derivative(x, x0, k):
#    f = np.exp(-k*(x-x0))
#    return -k / f
    
#def sigmoid_derivative(x, x0, k):
#	y = (k*np.exp(k*(x-x0)))/((np.exp(k*(x-x0))+1)*(np.exp(k*(x-x0))+1))
#	return y
    
with open('datapoint.csv') as csvfile:
    y_in = list(csv.reader(csvfile))
y_all = [[float(j) for j in i] for i in y_in]
y_trans = [list(k) for k in zip(*y_all)]
ydata = y_trans[2]

#y0_min = y_trans[0][0]
#y0_max = y_trans[0][len(y_trans[0])-1]

xdata = list(range(1, len(ydata)+1))

#numpy polyfit
#z = np.polyfit(xdata, ydata, 3)
#f = np.poly1d(z)

popt, pcov = curve_fit(sigmoid, xdata, ydata)

x = np.linspace(-1, 800, 1000)

#y = f(x)

y = sigmoid(x, *popt)

#y = logarithmic(x, popt[0], popt[1])

#print (sigmoid_derivative(100, *popt))
#newx = np.array([0, 100])
#newy = np.array([0, 0.843228521863])


#print(len(y))


print(popt[1])


#print(pcov)

#plt.plot(newx, newy,'g-', label='data')
plt.plot(xdata, ydata,'b-', label='data')
plt.plot(x,y, 'r--', linewidth=3, label='fit')
#fit = plt.plot(x,y, 'r--', linewidth=3, label='fit')
#fit_xy = fit[0].get_xydata()
#print(fit_xy[100])

plt.xlabel('time')
plt.ylabel('density')
plt.legend()
#plt.show()
plt.savefig('images/test.png')