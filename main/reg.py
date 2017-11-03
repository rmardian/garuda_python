import matplotlib
import matplotlib.pyplot as plt
import csv

#read features data
with open('3d-label-complete-424.csv') as csvfile:
    x = list(csv.reader(csvfile))
features = [[float(j) for j in i] for i in x]
features_transpose = [list(k) for k in zip(*features)]

fig, ax = plt.subplots(2, 2)

ax[0, 0].scatter(features_transpose[0], features_transpose[1])
ax[0, 0].set_title('Growth vs Lag Time')
ax[0, 0].set_xlabel('Growth')
ax[0, 0].set_ylabel('Lag Time')

ax[0, 1].scatter(features_transpose[1], features_transpose[2])
ax[0, 1].set_title('Lag Time vs Max OD')
ax[0, 1].set_xlabel('Lag Time')
ax[0, 1].set_ylabel('Max OD')

ax[1, 0].scatter(features_transpose[0], features_transpose[2])
ax[1, 0].set_title('Growth vs Max OD')
ax[1, 0].set_xlabel('Growth')
ax[1, 0].set_ylabel('Max OD')

plt.show()

#read label data
