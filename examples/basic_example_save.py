import sys

import numpy as np

sys.path.append('..')
from deploy import Component, Pipeline


inp_image = np.random.uniform(0, 1, size=(256, 256))

# Let's say the two functions I want to use are demo1 and demo2 from the function library

# Create the first Component
comp1 = Component(name='demo1', func='demo1')
# From the definition of demo1, we need to pass inp_image, add_value (optional), and mul_value (optional)
# We can pass these as Static Component variables since they are not needed elsewhere.
comp1.static_args = dict(inp_image=inp_image, add_value=10, mul_value=100)
# We want to get the final value of `inp_image` since that will need to be passed to demo2. So, record it.
comp1.to_record = dict(inp_image='inp_image')

# Create the second Component
comp2 = Component(name='demo2', func='demo2')
# From the definition of demo1, we need to pass inp_image and threshold_value. The former is a
# runtime argument that comes from the output of demo1, and threshold_value is a static argument.
comp2.static_args = dict(threshold_value=0.5)
comp2.runtime_args = dict(inp_image=comp1.FutureVariable.inp_image)
# i.e Get the value, at that time in the future when this function is executed,
# stored by name `inp_image` in comp1 and pass it to comp2's function (i.e demo2)
# as the kwarg value for `inp_image`
def dummy_save(var):
  print('I am a dummy save method; will not save anything. But I confirm to have received the variable {}'.format(var))
comp2.to_save = dict(inp_image='_aux/test_inp_image.npy', dummy_variable=dummy_save)

# We do not want to record anything for this component since we don't need any output from it.

# Create a pipeline and add these components
pipeline = Pipeline(name='demo')
pipeline.compile_pipeline([comp1, comp2])
# Run
pipeline.run()
