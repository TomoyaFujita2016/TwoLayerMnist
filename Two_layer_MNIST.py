import numpy as np
import matplotlib.pyplot as pyplot
import pickle as pk
import os
ex = lambda x: np.exp(np.clip(x, -709, 709))
sigmoid = lambda x: 1 / (1 + ex(np.clip(-x,-709,709)))

def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


with open("mnist.pkl", "rb") as f:
    X = pk.load(f, encoding = "latin1")
X0 = X[0][0]
p_Xt = X[0][1]
o_Xt = np.zeros((50000, 10))
for lines in range(np.shape(X[0][1])[0]):
    for n_Xt in range(10):
        if p_Xt[lines] == n_Xt:
            o_Xt[lines][n_Xt] = 1.0
        if o_Xt[lines][n_Xt] == 1.0:
            break
lr = 0.01
W0 = np.random.rand(200, 784) * 0.005
W1 = np.random.rand(10, 200) * 0.005
b = np.ones(2,)
X1 = np.zeros((50000, 200))
Y = np.zeros((50000, 10))
OUTPUT = np.zeros((50000,))
k_dW0 = np.zeros(10,)

dW0 = np.zeros_like(W0)
dW1 = np.zeros_like(W1)
dif = 0
for ep in range(100):
    print("ep="+str(ep))
    for rd in range(50000):
        #if rd % 5000 == 0:
        # print("Round="+str(rd))
            #print(dif)
        #print("  W0[0][160]="+str(W0[0][160]))
        #print("  W1[0][0]="+str(W1[0][0]))
        #=====Calculate OUTPUT
        X1[rd] = sigmoid(np.dot(X0[rd], W0.T) + b[0])
        Y[rd] = softmax(np.dot(X1[rd], W1.T) + b[1])
        #OUTPUT[rd] = np.argmax(Y[rd])

        old_Wb = np.sqrt( (np.sqrt(np.sum(W0 * W0)) **2) + (np.sqrt(np.sum(W1 * W1)) **2) + (np.sqrt(np.sum(b * b)) **2) )
        
        #=====prepare Update W0 & b0
        
        dW0 = np.dot( X0[rd][:, np.newaxis], ( np.dot( (Y[rd] - o_Xt[rd]),W1 ) * ( (1 - X1[rd]) * X1[rd] ) )[:, np.newaxis].T )
        #db0 = np.dot( (Y[rd] - o_Xt[rd]),W1 ) * ( (1 - X1[rd]) * X1[rd] )  
        
        #=====prepare Update W0 & b1
        
        dW1 =  np.dot(X1[rd][:, np.newaxis], (Y[rd] - o_Xt[rd])[:, np.newaxis].T )
        #db1 =  Y[rd] - o_Xt[rd]
        
        #=====Update
        W0 -= lr*dW0.T
        W1 -= lr*dW1.T
        #b[0] -= lr*np.sum(db0) #??
        #b[1] -= lr*np.sum(db1) #??


        #! may not converge
        #======Break
        new_Wb = np.sqrt( (np.sqrt(np.sum(W0 * W0)) **2) + (np.sqrt(np.sum(W1 * W1)) **2) + (np.sqrt(np.sum(b * b)) **2) )
        #print("Difference = "+str(np.absolute(new_Wb - old_Wb))) 
        dif = np.absolute(new_Wb - old_Wb) 
        if dif < 1e-10:
            break
      
    with open('Learned_W0.txt', 'wb') as f:
        pk.dump(W0, f)
    with open('Learned_W1.txt', 'wb') as f:
        pk.dump(W1, f)
    os.system("python test_Wl.py")

with open('Learned_W0.txt', 'wb') as f:
    pk.dump(W0, f)
with open('Learned_W1.txt', 'wb') as f:
    pk.dump(W1, f)
print("Finished!!")
