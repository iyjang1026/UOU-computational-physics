#step01.py

import numpy as np 


class Variable: #함수들의 묶음
    def __init__(self,data): #method(함수 같음) => 함수와 데이터가 클래스에 같이 있음. __init__의 역할: 클래스를 이용해서 객체를 생성. 객체를 생성하는 함수
        self.data = data #self:인스턴스 변수, 인스턴스 변수로서, self를 class안에서 범용적으로 사용가능
    #self를 data로 반황하게 함. 
class Function:
    def __call__(self, input): #__init__와 비슷하지만, class로 할 작업을 정의함.
        x = input.data
        y = x**2
        output = Variable(y)
        return output #일정 return값으로 return함. __init__는 그냥 작업을 반환.
    
if __name__ == "__main__":
    import numpy as np
    x = Variable(np.array(10))
    f = Function()
    y = f(x)


    print(type(y))
    print(y.data)
