#step04_2.py
import numpy as np
from step03 import Function, Square, Variable, Exp

def num_diff_forward(func, x, eps=1e-4):
    """전진차분
    f'(x)= (f(x+eps)-f(x))/eps"""
    x0 = Variable(x.data)
    x1 = Variable(x.data + eps)
    y0 = func(x0)
    y1 = func(x1)
    return (y1.data - y0.data) / eps

def num_diff_center(func, x, eps=1e-4):
    """중앙차분
    f'(x) = (f(x+eps)-f(x-eps))/eps"""
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = func(x0)
    y1 = func(x1)
    return (y1.data - y0.data) / (2*eps)

if __name__ == "__main__":
    def f(x):
        A = Square(); B = Exp() 
        return A(B(A(x)))
    x = Variable(np.array(0.5))
    dy = num_diff_center(f, x)
    print(dy)