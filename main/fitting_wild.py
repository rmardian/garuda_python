import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

def sigmoid (x, x0, k):
	y = 1 / (1 + np.exp(-k*(x-x0)))
	#y = 1 / (1 + bigfloat.exp(-k*(x-x0),bigfloat.precision(100)))
	return y

def time_convert (t):
	(h, m, s) = t.split(':')
	return (int(h) * 3600 + int(m) * 60 + int(s))
 
def find_y (m, y1, x1, x):
	return (m * (x - x1)) + y1
     
#def sigmoid_derivative(x, x0, k):
#    f = np.exp(-k*(x-x0))
#    return -k / f
    
#def sigmoid_derivative(x, x0, k):
#	y = (k*np.exp(k*(x-x0)))/((np.exp(k*(x-x0))+1)*(np.exp(k*(x-x0))+1))
#	return y
    
with open('datapoint001.csv') as csvfile:
    y_in = list(csv.reader(csvfile))
    
y_all = [[float(j) for j in i] for i in y_in]
y_trans = [list(k) for k in zip(*y_all)]


#read label data
with open('time.csv') as csvfile:
    x_in = list(csv.reader(csvfile))
time_temp = [[j for j in i] for i in x_in]
time = [val for sublist in time_temp for val in sublist]
#print label

time_dec = [time_convert(t) for t in time]
#print (time_dec)


for i in range(len(y_trans)):
	
	try:
	
		ydata = y_trans[i]
		xdata = list(range(1, len(ydata)+1))

		#popt, pcov = curve_fit(sigmoid, xdata, ydata)

		#x = np.linspace(time_dec[0], time_dec[len(time_dec)-1], 100000)
		#y = sigmoid(x, *popt)

		#print(str(i) + ',' + str(popt[1]))
		print(str(i) + ',')
		
		#xa = np.linspace(0,10)
		#ya = x*2

		#p1 = [1,20]
		#p2 = [6,70]

		#plt.plot(xa, ya)
		#newline(p1,p2)

		plt.plot(time_dec, ydata,'b-', label='data')
		#plt.plot(x,y, 'r--', linewidth=3, label='fit')
		#plt.xticks(x, time)
		plt.xlabel('time')
		plt.ylabel('density')
		plt.ylim([0, 1])
		plt.legend()
		#plt.show()
		plt.savefig('images_plate_1/plate1_' + str(i) + '.png')
		plt.clf()
		
	except:
		print(str(i) + ',error')
		pass