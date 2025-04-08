from contextlib import contextmanager
import numpy as np

from step16 import Variable, Square

class Config:
    enable_bakcprop = True
    
@contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try :
        yield #예외처리
    finally :
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

with no_grad():
    x = Variable(np.array(2.0))
    y = Square(x)
