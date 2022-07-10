import matplotlib.pyplot as plt
import numpy as np 

import pdb


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10, 7.2))

plt.grid(ls="dotted",b=True,axis='y')
plt.xlim((8, 92))

x = np.linspace(10, 90, 9)
plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],fontsize=10)

plt.ylim((24,76))
y = np.linspace(25, 75, 11)
plt.yticks(y, fontsize=10)

baseline = [71.737] * 9

plt.plot(x, baseline, linewidth=2, color = '#9B9B7A', label='Origin')

unstructure = [71.95, 71.89, 71.81, 71.67, 71.15, 70.15, 68.57, 65.85, 59.73]
structure = [71.55, 70.87, 69.94, 68.33, 66.73, 60.65, 54.77, 45.82, 28.04]
block2 = [71.616, 71.449, 71.277, 70.853, 70.233, 68.690, 67.229, 64.403, 58.131]
block4 = [71.504, 71.373, 70.936, 70.458, 69.426, 68.314, 66.426, 62.745, 56.667]
block8 = [71.510, 71.192, 70.819, 70.171, 69.372, 68.074, 66.207, 62.315, 55.756]
block16 = [71.433, 71.092, 70.697, 70.022, 69.352, 67.811, 65.272, 61.126, 53.005]
block32 = [71.419, 71.024, 70.496, 69.79, 68.762, 67.381, 63.634, 58.962, 51.26]

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
plt.savefig("mobilenetv2.pdf",bbox_inches='tight')
plt.show()