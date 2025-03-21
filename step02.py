#step02.py
from step01 import Variable
import numpy as np
class Function:
    def __call__(self, input): #__init__와 비슷하지만, class로 할 작업을 정의함.
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output #일정 return값으로 return함. __init__는 그냥 작업을 반환.
    
    def forward(self, x): #forward의 종류에 따라 출력을 바꿈. 상속. forward를 특정 함수로 출력하게 함.
        raise NotImplementedError() #forward가 정의 되지 않으면 error를 출력
    
class Square(Function): #class 밖에서도 다양하게 method를 정의할 수 있음.
    def forward(self, x):
        return x**2
    
if __name__ == "__main__":
    x = Variable(np.array(range(0,10)))
    f = Square()
    y = f(x)
    
    print(type(y))
    print(y.data)
    
