import numpy as np
import time

from deploy import Pipeline, Component

inp_image = np.zeros((4,4))

def my_func1(x):
    print('I am func1. I received: ', x)
    x = x +5
    return locals()

def my_func2(x):
    print('I am func2. I received: ', x)
    x = x + 10
    time.sleep(20)
    return locals()

def my_func3(x):
    print('I am func3. I received: ', x)
    x = x + 2
    return locals()

comp1 = Component('comp1', func=my_func1)
comp1.static_args = dict(x=inp_image)
comp1.to_record = dict(x='x')

comp2 = Component('comp2', func=my_func2, debug=True)
comp2.runtime_args = dict(x=comp1.FutureVariable.x)

comp3 = Component('comp3', func=my_func3, debug=True)
comp3.runtime_args = dict(x=comp1.FutureVariable.x)

pipeline = Pipeline('my_pipeline')
pipeline.compile_pipeline([comp1, comp2, comp3], parallelize=True)

pipeline.run()
