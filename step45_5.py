import numpy as np
import cpnn.functions as F
from cpnn.models import MLP
from cpnn import Variable
import matplotlib.pyplot as plt

np.random.seed(0)

x = Variable(np.random.rand(100,1))
y = F.sin(2 * np.pi * x) + np.random.rand(100,1)
lr = 0.2
max_iter = 10000
model = MLP((10,1))

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)