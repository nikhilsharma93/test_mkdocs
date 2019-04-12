import sys

import numpy as np

sys.path.append('..')
from deploy import Component, Pipeline


inp_image = np.random.uniform(0, 1, size=(256, 256))

# Initialize a pipeline, and set the pipeline variable `add_value`
pipeline = Pipeline(name='demo')
pipeline.add_pipeline_variable(dict(add_value=5))

comp1 = Component(name='demo1', func='demo1')
comp1.static_args = dict(inp_image=inp_image, mul_value=100)
comp1.runtime_args = dict(add_value=pipeline.FutureVariable.add_value)
comp1.to_record = dict(inp_image='inp_image')

comp3 = Component(name='demo3', func='demo3')
comp3.runtime_args = dict(inp_image=comp1.FutureVariable.inp_image, add_value=pipeline.FutureVariable.add_value)

# Note that `compile_pipeline` should be the last operation performed on the pipeline before calling the `.run` on it
pipeline.compile_pipeline([comp1, comp3])
pipeline.run()
