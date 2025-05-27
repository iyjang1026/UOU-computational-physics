import numpy as np
from cpnn import Variable, optimizers
import cpnn.functions as F
from cpnn.models import MLP
import matplotlib.pyplot as plt

np.random.seed(0)

x = np.random.rand(100,1)
y = np.sin(2 * np.pi *x) + np.random.rand(100,1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizers = optimizers.MomentumSGD(lr)
optimizers.setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizers.update()
    if i % 1000 == 0:
        print(loss)

x1 = Variable(np.linspace(0,1,100).reshape(100,1))
y1 = model(x1)

plt.scatter(x, y)
plt.plot(x1.data, y1.data, c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()