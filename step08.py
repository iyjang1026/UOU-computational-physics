import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs: #반복문으롤 변경
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
        gx = 2 * x * gy 
        return gx

class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx

if __name__ == "__main__":
    x = Variable(np.array(0.5))
    A = Square()
    B = Exp()
    C = Square()
    y = C(B(A(x)))
    y.grad = np.array(1.0)

    y.backward()
    #print(f"{x.grad:.4f}")