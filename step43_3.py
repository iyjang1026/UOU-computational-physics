import numpy as np
from cpnn import Variable
import cpnn.functions as F
import matplotlib.pyplot as plt

np.random.seed(0)
x = Variable(np.random.rand(100,1))
y = F.sin(2*np.pi *x) + np.random.rand(100,1)
x0 = np.arange(-5,5,0.1)
y0 = 1 / 1+np.exp(-x0)
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H,O))
b2 = Variable(np.zeros(O))

def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:
        print(loss)

x1 = Variable(np.linspace(0,1,100).reshape(100,1))
y1 = predict(x1)

plt.scatter(x.data, y.data)
plt.plot(x1.data, y1.data, c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()