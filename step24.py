import cpnn.core_simple as cs
import numpy as np

def sphere(x0, x1):
    z = x0**2 + x1**2
    return z
def matyas(x0, x1):
    z = 0.26 * (x0 ** 2 + x1 ** 2) - 0.48 * x0 * x1
    return z

x = cs.Variable(np.array(1.0))
y = cs.Variable(np.array(1.0))

z = matyas(x, y)

z.backward()

print(x.grad, y.grad)