#step03.py
import numpy as np
from step02 import Function, Square, Variable

class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    
if __name__ == "__main__":
    x = Variable(np.array(1.0))
    A = Square()
    B = Exp()
   
    a = A(x)
    b = B(x)
    y = A(b)


    print(type(y))
    print(y.data)
