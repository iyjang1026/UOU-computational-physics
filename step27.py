"""
STEP 27
테일러 급수 미분
"""
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math
from cpnn import Variable
from cpnn import Function
from cpnn.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


if __name__ == "__main__":
    x = Variable(np.array(np.pi/4))
    y = sin(x)
    y.backward()
    print(y.data)
    print(x.grad)
    plot_dot_graph(y, verbose=False, to_file='/Users/jang-in-yeong/UOU-computational-physics/sin.png')

    x = Variable(np.array(np.pi/4))
    y = my_sin(x)
    y.backward()
    print(y.data)
    print(x.grad)
    plot_dot_graph(y, verbose=False, to_file='/Users/jang-in-yeong/UOU-computational-physics/my_sin.png')
