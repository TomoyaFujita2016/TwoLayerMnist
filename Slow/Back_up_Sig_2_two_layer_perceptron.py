import numpy as np
import matplotlib.pyplot as pyplot
import pickle as pk

ex = lambda x: np.exp(np.clip(x, -709, 709))
sigmoid = lambda x: 1 / (1 + ex(np.clip(-x,-709,709)))
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
for rd in range(50):
    print("Round="+str(rd))
    #print("  W0[0][160]="+str(W0[0][160]))
    #print("  W1[0][0]="+str(W1[0][0]))
    #=====Calculate OUTPUT
    X1[rd] = sigmoid(np.dot(X0[rd], W0.T) + b[0])
    Y[rd] = sigmoid(np.dot(X1[rd], W1.T) + b[1])
    #OUTPUT[rd] = np.argmax(Y[rd])
  
    #=====Update b1
    for m in range(10):
        for n in range(200):
            tmp2 = np.dot(W1[m], X1[rd].T)
            dW1 = ex(tmp2) * (sigmoid(tmp2) ** 2)
            b[1] -= lr*-dW1*(o_Xt[rd][m] - Y[rd][m])
    
    #=====Update b0
        
    for m in range(200):
        for n in range(784):
            for k in range(10):
                #print ("m="+str(m)+"  n="+str(n)+"  k="+str(k))
                tmp1_1 = np.dot(X1[rd], W1[k].T) + b[1]
                #if tmp1_1 < -100:
                    #tmp1_1 = 0
                tmp1 = ex(tmp1_1) / (sigmoid(tmp1_1)**2)
                
                tmp2_1 = np.dot(X0[rd], W0[m].T) + b[0]
                #if tmp2_1 < -100:
                    #tmp2_1 = 0
                tmp2 = ex(tmp2_1) / (sigmoid(tmp2_1)**2)
                k_dW0[k] = -W1[k][m] * tmp1 * tmp2 * (o_Xt[rd][k] - Y[rd][k])
            dW0 = np.sum(k_dW0)
            b[0] -= lr*-dW0


    
    #=====Update W1
    for m in range(10):    
        for n in range(200):
           
            tmp1_1 = np.dot(W0[n], X0[rd].T )
            #if tmp1_1 < -100:
                #tmp1_1 = 0
            tmp1 = sigmoid(tmp1_1)
           
            tmp2 = np.dot(W1[m], X1[rd].T)
            #if tmp2 < -100:
                #tmp2 = 0
                    
            dW1 = ex(tmp2) * (sigmoid(tmp2) ** 2) * tmp1
            W1[m][n] -= lr*-dW1*(o_Xt[rd][m] - Y[rd][m])
    #=====Update W0
    for m in range(200):
        for n in range(784):
            for k in range(10):
                #print ("m="+str(m)+"  n="+str(n)+"  k="+str(k))
                tmp1_1 = np.dot(X1[rd], W1[k].T) + b[1]
                #if tmp1_1 < -100:
                    #tmp1_1 = 0
                tmp1 = ex(tmp1_1) / (sigmoid(tmp1_1)**2)
                
                tmp2_1 = np.dot(X0[rd], W0[m].T) + b[0]
                #if tmp2_1 < -100:
                    #tmp2_1 = 0
                tmp2 = ex(tmp2_1) / (sigmoid(tmp2_1)**2)
                
                k_dW0[k] = -W1[k][m] * X0[rd][m] * tmp1 * tmp2 * (o_Xt[rd][k] - Y[rd][k])
            dW0 = np.sum(k_dW0)
            W0[m][n] -= lr*dW0


with open('Learned_W0.txt', 'wb') as f:
    pk.dump(W0, f)
with open('Learned_W1.txt', 'wb') as f:
    pk.dump(W1, f)
print("Finished!!!!!")
