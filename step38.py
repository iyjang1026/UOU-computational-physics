import numpy as np
from cpnn import Variable
import cpnn.functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

x = Variable(np.random.randn(1, 2, 3))
y = x.reshape((2, 3))
print(y)
y = x.reshape([2, 3])
print(y)
y = x.reshape(2, 3)
print(y)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.rand(2, 3))
y = x.transpose()
print(y)
y = x.T
print(y)

x = Variable(np.random.rand(1, 2, 3, 4))
y = x.transpose(1, 0, 3, 2)
print(y)
y = x.T
print(y)
