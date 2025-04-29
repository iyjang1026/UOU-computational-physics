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

if __name__ == "__main__":
    x = Variable(np.array(np.pi/4))
    y = sin(x)
    y.backward()
    print(y.data)
    print(x.grad)
    plot_dot_graph(y, verbose=False, to_file='/Users/jang-in-yeong/UOU-computational-physics/sin.png')