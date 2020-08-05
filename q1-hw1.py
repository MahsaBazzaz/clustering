import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
d = pd.read_csv('dataset.csv')
X1=d['X1']
X2=d['X2']
label=d['Label']
scatter_x = np.array(X1)
scatter_y = np.array(X2)
group = np.array(label)
cdict = {0: 'red', 1: 'blue'}
fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    #label=0 -> red label=1 ->blue
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 6)
ax.legend()
plt.show()
