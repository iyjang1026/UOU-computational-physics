"""
STEP 29
뉴턴 방법으로 푸는 최적화(수동 계산)
"""
import numpy as np
from cpnn import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(np.array(2.0))
lr = 0.001
iters = 100

for i in range(iters):
    y = f(x)

    x.cleargrad()
    y.backward()

    x.data -= lr * x.grad
print(f"경사하강법> # of iterations: {iters:3},   최소값에서의 x: {x.data:8.4f}")

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)

print(f"뉴튼법    > # of iterations: {iters:3},   최소값에서의 x: {x.data:8.4f}")
