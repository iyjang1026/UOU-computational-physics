import numpy as np

class Variable:
    def __init__(self, data):
        if data != None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not surported.")
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    def backward(self):
        if self.grad == None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            x.backward()
            if x.creator != None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError
    def backward(self, x):
        raise NotImplementedError
    
class Square(Function):
    def forward(self, x):
        return x**2
    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy 
        return gx

class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx
    
if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array([1.0]))
    y = C(B(A(x)))
    y.grad = np.array(1.0)

    y.backward()
    print(F"{x.grad:.4f}")