import numpy as np
import pickle as pk

def sigmoid(x):
    return 1/(1 + np.exp(-x))


with open('poor_Learned_W0.txt', 'rb') as f:
    W0 = pk.load(f, encoding='latin1')
with open('poor_Learned_W1.txt', 'rb') as f:
    W1 = pk.load(f, encoding='latin1')
with open('mnist.pkl', 'rb') as f:
    X = pk.load(f, encoding='latin1')
    X0 = X[1][0]
    Xt = X[1][1]
OUTPUT = np.zeros((50000,))
b = np.ones((2,))
X1 = np.zeros((50000, 200))
Y = np.zeros((50000, 10))
suc = 0
for rd in range(10000):
    #rd = np.random.randint(10000)
    X1[rd] = sigmoid(np.dot(X0[rd], W0.T) + b[0])
    Y[rd] = sigmoid(np.dot(X1[rd], W1.T) + b[1])
    OUTPUT[rd] = np.argmax(Y[rd])
    print("ANS=" + str(OUTPUT[rd]) + "\nOUTPUT=" + str(Xt[rd]) + "\n")
    if OUTPUT[rd] == Xt[rd]:
        suc += 1
    
acc = suc / 10000

print("Accuracy="+str(acc))
