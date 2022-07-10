import matplotlib.pyplot as plt
import numpy as np 

import pdb


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10, 7.2))

plt.grid(ls="dotted",b=True,axis='y')
plt.xlim((8, 92))

x = np.linspace(10, 90, 9)
plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],fontsize=10)

plt.ylim((65,78))
y = np.linspace(66, 78, 7)
plt.yticks(y, fontsize=10)

baseline = [77.008] * 9

plt.plot(x, baseline, linewidth=2, color = '#9B9B7A', label='Origin')

unstructure = [77.334, 77.404, 77.152, 77.086, 77.088, 77.062, 76.846, 75.950, 72.772]
structure = [76.944, 76.930, 76.494, 75.794, 75.382, 74.338, 73.430, 70.958, 66.960]
block2 = [77.254, 77.214, 77.246, 77.202, 76.654, 76.442, 75.906, 74.604, 71.930]
block4 = [77.502, 77.102, 77.084, 76.890, 76.506, 76.170, 75.410, 73.742, 70.028]
block8 = [77.142, 77.032, 76.794, 76.706, 76.146, 76.034, 75.042, 73.790, 70.690]
block16 = [77.190, 76.920, 76.814, 76.496, 76.254, 75.542, 74.688, 73.336, 70.016]
block32 = [77.046, 77.032, 76.716, 76.354, 75.960, 75.566, 74.606, 72.842, 68.942]

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
plt.title('Top-1',fontsize = 20)
plt.legend(fontsize=20,loc="lower left")
plt.savefig("resnet50-top1.pdf",bbox_inches='tight')
plt.show()