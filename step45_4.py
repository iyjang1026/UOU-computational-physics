import numpy as np
from cpnn import Variable, Model
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
max_iters = 10000
hidden_size = 10

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

model = TwoLayerNet(hidden_size, 1)

for i in range(max_iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

x1 = Variable(np.linspace(0,1,100).reshape(100,1))
y1 = model(x1)

plt.scatter(x.data, y.data)
plt.plot(x1.data, y1.data, c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()