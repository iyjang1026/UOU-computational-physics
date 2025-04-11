import numpy as np
"""
class에 연산자 추가
"""
class Variable:
    def __init__(self, data):
        self.data = data

    def __add__(self, other): #variable의 덧셈
        return Add()(self, other)
    def __radd__(self, other):
        return Add()(self, other)
    def __mul__(self, other):
        return Mul()(self, other)
    def __rmul__(self, other):
        return Mul()(self, other)
class Add:                      #덧셈
    def __call__(self, x0, x1):
        if not isinstance(x0, Variable): #입력을 전부 Variable로 변환
            x0 = Variable(x0)
        if not isinstance(x1, Variable):
            x1 = Variable(x1)

        y = x0.data + x1.data
        return Variable(y)
    
class Mul:                      #곱셈
    def __call__(self, x0, x1):
        if not isinstance(x0, Variable):
            x0 = Variable(x0)
        if not isinstance(x1, Variable):
            x1 = Variable(x1)    
        y = x0.data * x1.data
        return(Variable(y))
    

x0 = Variable(3)


