import matplotlib.pyplot as plt
import numpy as np 

import pdb


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10, 7.2))
plt.grid(ls="dotted",b=True,axis='y')
plt.xlim((8, 92))

x = np.linspace(10, 90, 9)
plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],fontsize=10)

plt.ylim((86,94.2))
y = np.linspace(87, 94, 8)
plt.yticks(y, fontsize=10)

baseline = [93.654] * 9


plt.plot(x, baseline, linewidth=2, color = '#9B9B7A', label='Origin')

unstructure = [93.632, 93.630, 93.656, 93.724, 93.614, 93.558, 93.510, 93.000, 91.330]
structure = [93.348, 93.324, 93.100, 92.782, 92.518, 92.102, 91.466, 90.206, 87.576]
block2 = [93.596, 93.674, 93.560, 93.550, 93.466, 93.296, 93.044, 92.258, 90.780]
block4 = [93.590, 93.498, 93.492, 93.488, 93.238, 93.042, 92.734, 91.850, 89.870]
block8 = [93.528, 93.562, 93.480, 93.316, 93.134, 92.928, 92.480, 91.684, 89.988]
block16 = [93.512, 93.534, 93.402, 93.112, 93.084, 92.824, 92.290, 91.538, 89.510]
block32 = [93.602, 93.410, 93.322, 93.210, 92.950, 92.698, 92.138, 91.198, 88.764]

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
plt.savefig("resnet50-top5.pdf",bbox_inches='tight')
plt.show()