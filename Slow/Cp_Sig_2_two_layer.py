# This is slower than other program.
# This program accesses element of W one by one. That's why.

import numpy as np
import matplotlib.pyplot as pyplot
import pickle as pk

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
for rd in range(2000):
    print("Round="+str(rd))
    #print("  W0[0][160]="+str(W0[0][160]))
    #print("  W1[0][0]="+str(W1[0][0]))
    #=====Calculate OUTPUT
    X1[rd] = sigmoid(np.dot(X0[rd], W0.T) + b[0])
    Y[rd] = softmax(np.dot(X1[rd], W1.T) + b[1])
    #OUTPUT[rd] = np.argmax(Y[rd])

    old_Wb = np.sqrt( (np.sqrt(np.sum(W0 * W0)) **2) + (np.sqrt(np.sum(W1 * W1)) **2) + (np.sqrt(np.sum(b * b)) **2) )
    
    #=====Update W0 & b0
    for i in range(200):
        for n in range(784):
            for k in range(10):
                if o_Xt[rd][k] == 1:
                    break

            tmp_a = (X0[rd][n] * W0[i][n])
            tmp_b = sigmoid(tmp_a)
            tmp_c = np.sum( np.dot(X1[rd], W1.T) + b[1]) - (np.dot(X1[rd], W1.T) + b[1])[k]

            dZW0 = (-tmp_c*ex(-tmp_b) / ( 1 + tmp_c*ex(-tmp_b))) * (ex(-tmp_a) / ((1 + ex(-tmp_a))**2)) * X0[rd][n] * W1[k][i]
            dZb0 = (-tmp_c * ex(-tmp_b) / ( 1 + tmp_c * ex(-tmp_b))) * (tmp_c * ex(-tmp_a) / ((1 + ex(-tmp_a))**2)) * W1[k][i]

            W0[i][n] -= lr * dZW0
            b[0] -= lr * dZb0

            #print(dZW0)
            #print("\n")
            #print(dZb0)
    #=====Update W0 & b1
    for k in range(10):
        for j in range(200):
            tmp_a = (np.dot(X1[rd], W1.T) + b[1])[k]
            tmp_b = (sigmoid(np.dot(X0[rd], W0.T) + b[0]))[j]
            tmp_c = np.sum( np.dot(X1[rd], W1.T) + b[1]) - (np.dot(X1[rd], W1.T) + b[1])[k]

            dZW1 = (-(tmp_c * ex(-tmp_a)) / (1 + (tmp_c * ex(-tmp_a)))) * tmp_b
            dZb1 = (-(tmp_c * ex(-tmp_a)) / (1 + (tmp_c * ex(-tmp_a)))) 

            W1[k][j] -= lr * dZW1
            b[1] -= lr * dZb1

    #======Break
    new_Wb = np.sqrt( (np.sqrt(np.sum(W0 * W0)) **2) + (np.sqrt(np.sum(W1 * W1)) **2) + (np.sqrt(np.sum(b * b)) **2) )
    #print("Difference = "+str(np.absolute(new_Wb - old_Wb))) 
    if np.absolute(new_Wb - old_Wb) < 0.001:
        break


with open('poor_Learned_W0.txt', 'wb') as f:
    pk.dump(W0, f)
with open('poor_Learned_W1.txt', 'wb') as f:
    pk.dump(W1, f)
print("Finished!!!!!")
