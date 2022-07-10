import matplotlib.pyplot as plt
import numpy as np 

import pdb


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10, 7.2))

plt.grid(ls="dotted",b=True,axis='y')
plt.xlim((8, 92))

x = np.linspace(10, 90, 9)
plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],fontsize=10)

plt.ylim((48,92))
y = np.linspace(50, 90, 9)
plt.yticks(y, fontsize=10)

baseline = [90.452] * 9


plt.plot(x, baseline, linewidth=2, color = '#9B9B7A', label='Origin')

unstructure = [90.50, 90.48, 90.38, 90.21, 89.87, 89.42, 88.39, 86.50, 81.91]
structure = [90.20, 89.91, 89.28, 88.29, 87.19, 82.85, 78.49, 70.97, 52.34]
block2 = [90.270, 90.085, 90.015, 89.840, 89.417, 88.668, 87.512, 85.300, 80.377]
block4 = [90.103, 90.065, 89.832, 89.603, 88.899, 88.153, 86.957, 83.966, 79.163]
block8 = [90.165, 89.957, 89.728, 89.340, 88.862, 87.926, 86.521, 83.480, 78.528]
block16 = [90.077, 90.009, 89.726, 89.320, 88.708, 87.658, 86.005, 82.884, 76.278]
block32 = [90.124, 89.800, 89.565, 89.113, 88.425, 87.456, 84.758, 81.228, 74.490]

plt.plot(x, unstructure, linewidth=2, marker='*', markersize=8, color = '#7b2cbf', label='Weight')
plt.plot(x, structure, linewidth=2, marker='D', color = '#F3722C', label='Filter')
plt.plot(x, block2, linewidth=2, marker='o', color = '#ff5d8f',label='Block 1x2')
plt.plot(x, block4, linewidth=2, marker='v', color = '#90BE6D',label='Block 1x4')
plt.plot(x, block8, linewidth=2, marker='^', color = '#F9C74F',label='Block 1x8')
plt.plot(x, block16, linewidth=2, marker='<', color = '#00afb9',label='Block 1x16')
plt.plot(x, block32, linewidth=2, marker='>', color = '#43AA8B',label='Block 1x32')

plt.xticks(x, fontsize=15)
plt.yticks(y, fontsize=15)

plt.xlabel(r'Pruning Rate $p$',fontsize = 20)
plt.ylabel('Accuracy (%)',fontsize = 20)
plt.title('Top-5',fontsize = 20)
plt.legend(fontsize=20,loc="lower left")
plt.savefig("mobilenetv2-top5.pdf",bbox_inches='tight')
plt.show()