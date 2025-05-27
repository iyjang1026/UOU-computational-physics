import numpy as np
from cpnn import Variable
import cpnn.functions as F
import cpnn.layers as L
import matplotlib.pyplot as plt

np.random.seed(0)
x = Variable(np.random.rand(100,1))
y = F.sin(2*np.pi *x) + np.random.rand(100,1)

l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1,l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

x1 = Variable(np.linspace(0,1,100).reshape(100,1))
y1 = predict(x1)

plt.scatter(x.data, y.data)
plt.plot(x1.data, y1.data, c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()