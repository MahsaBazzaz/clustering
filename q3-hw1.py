from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
#sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
#dcost/dW as calculated in question 2
def compute_dW_cost(yT,y,x):
	return (y-yT)*x
#dcost/db as calculated in question 2
def compute_dB_cost(yT,y):
	return (y-yT)
#load datas
X1,X2,Label = np.loadtxt('train-set.csv',unpack=True,delimiter=',')
n = 125 #number of train datas
lr = 12.8 #learning rate
b = 0.0   #bias
w = np.array([[1.0],[1.0]]) #weights
n_epoch = 7535 #number of iterations

for i in range(0,n_epoch):
    #for each W or n grad[W]=0 and grad[b]=0
	gradW = np.array([[0.0],[0.0]])
	gradB = 0.0
	for k in range(0,n):
        #compute y: y=S(W.X+b)
		y = sigmoid((w[0]*X1[k])+ (w[1]*X2[k])+ b)
		#because we already have(dcost/dW) we do not calculate cost
		cost_dw1 = compute_dW_cost(Label[k],y,X1[k])
        #for each W grad[W]+=dcost/dW
		gradW[0] += cost_dw1
		cost_dw2 = compute_dW_cost(Label[k],y,X2[k])
		gradW[1] += cost_dw2 
		#for each b grad[b]+=dcost/db
		cost_dB = compute_dB_cost(Label[k],y)
		gradB += cost_dB
	#for each W W=W-(le * grad[W])/n	
	w[0] -= (lr * gradW[0])/n
	w[1] -=(lr * gradW[1])/n
    #for each b b=b-(le * grad[b])/n	
	b -= (lr * gradB)/n
	
s = 0.0 #Mean squared error (MSE)
correct = 0.0 #number of correct predection
#load datas
X1t,X2t,Labelt = np.loadtxt('train-set.csv',unpack=True,delimiter=',')
n2 = 55 #number of test datas
for i in range(0,n2):
    #compute y(predict): y=S(W.X+b)
    out = sigmoid((w[0]*X1t[i])+(w[1]*X2t[i])+b)
    #compute error (real value- predection)
    error = abs(Labelt[i] - out)
    if(error < 0.5):
        correct = correct + 1
        #plot correct predections with green
        plt.scatter(X1t[i], X2t[i], s=6, c='green', alpha=1)
    else:
        #plot correct predections with green
        plt.scatter(X1t[i], X2t[i], s=6, c='red', alpha=1)
    s += (out -Labelt[i]) * (out -Labelt[i])
#print MSE
print(s/n2)
#print Accuracy (number of correct prediction/number of samples)
print('Accuracy = ', (correct*100)/n2, ' %')
plt.ylabel('X2 axis')
plt.xlabel('X1 axis')
plt.show()    
    
        