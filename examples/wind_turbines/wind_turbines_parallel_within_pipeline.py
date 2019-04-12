"""Example showing the wind turbines model deployment in "parallel within pipeline" mode.
Usage: python wind_turbines_noparallel.py

Looking at `summary_graph.png` in this folder will give a good idea about the way this pipeline is structured.
"""

from datetime import datetime
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"] = "8"

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

sys.path.append('../..')
from deploy import Component, Pipeline, repeator
from local_func_library import get_keys, spinworker


startTime = datetime.now()

def main_func(keys, model_path):
    from datetime import datetime
    from functools import partial

    import numpy as np

    from appsci_utils.generator.transforms import cast_transform, rescale_feature_transform

    from local_func_library import (
        get_raster, load_model_func, loop_key_deployer, model_pred_wrapper,
        postprocess, preprocess, uploader)


    # Create some initial variables
    transforms = [partial(cast_transform, feature_type=np.float32, target_type=np.float32),
                  partial(rescale_feature_transform, shift=128.0, scale=128.0)]

    pipeline = Pipeline('pre_predict_post_upload')
    pipeline.add_pipeline_variable(repeator(products=['airbus:oneatlas:spot:v1']))
    pipeline.add_pipeline_variable(repeator(break_tilesize=1024, unpadded_tilesize=1024, pad=0))

    main_pipeline = Pipeline('main')

    d0_model_load = Component(name='load_model', func=load_model_func)
    d0_model_load.static_args = repeator(model_path=model_path)
    d0_model_load.to_record = repeator('model')

    # We will be parallelizing this `pre_predict_post_upload` pipeline.
    # Let's set the queue_size as 2 for each of these components.
    # Also, for reasons mentioned in the documentation, we will run the model prediction in the main process
    # but setting run_async=False for the predict component
    c0_raster = Component(name='get_raster', func=get_raster, queue_size=2)
    c0_raster.static_args = repeator(bands=['red', 'green', 'blue'], start_datetime='2014-01-01',
                                     end_datetime='2019-01-01')
    c0_raster.runtime_args = {'products': pipeline.FutureVariable.products}
    c0_raster.to_record = repeator('batch_imgs', 'batch_keys', 'batch_ids', 'batch_meta', 'batch_epsg')

    c1_preprocess = Component(name='preprocessing', func=preprocess, queue_size=2)
    c1_preprocess.static_args = repeator(transforms=transforms)
    c1_preprocess.runtime_args = {'batch_imgs': c0_raster.FutureVariable.batch_imgs}
    c1_preprocess.to_record = repeator('batch_imgs')

    # Just for fun, let us run the prediction step in debug mode
    c3_model_predict = Component(name='predict', func=model_pred_wrapper, queue_size=2,
                                 run_async=False, debug=True)
    c3_model_predict.runtime_args = {'batch_imgs': c1_preprocess.FutureVariable.batch_imgs,
                                     'break_tilesize': pipeline.FutureVariable.break_tilesize,
                                     'unpadded_tilesize': pipeline.FutureVariable.break_tilesize,
                                     'model': d0_model_load.FutureVariable.model,
                                     'pad': pipeline.FutureVariable.pad}
    c3_model_predict.to_record=repeator('batch_proba')

    c4_postprocess = Component(name='postprocess', func=postprocess, queue_size=2)
    c4_postprocess.static_args = repeator(upload_dtype='uint8')
    c4_postprocess.runtime_args = {'batch_ids': c0_raster.FutureVariable.batch_ids,
                                   'break_tilesize': pipeline.FutureVariable.break_tilesize,
                                   'unpadded_tilesize': pipeline.FutureVariable.unpadded_tilesize,
                                   'pad': pipeline.FutureVariable.pad,
                                   'batch_meta': c0_raster.FutureVariable.batch_meta,
                                   'batch_epsg': c0_raster.FutureVariable.batch_epsg,
                                   'batch_keys': c0_raster.FutureVariable.batch_keys,
                                   'batch_imgs': c3_model_predict.FutureVariable.batch_proba}
    c4_postprocess.to_record = repeator('batch_proba', 'image_ids', 'upload_kwargs')

    c5_uploader = Component(name='upload', func=uploader, queue_size=2)
    c5_uploader.runtime_args = {'batch_proba': c4_postprocess.FutureVariable.batch_proba,
                'image_ids': c4_postprocess.FutureVariable.image_ids,
                'upload_kwargs': c4_postprocess.FutureVariable.upload_kwargs,
                'products': pipeline.FutureVariable.products}

    pipeline.compile_pipeline([c0_raster, c1_preprocess, c3_model_predict, c4_postprocess, c5_uploader],
                              parallelize=True)


    d1_loopkeys = Component(name='loopkeys', func=loop_key_deployer)
    main_pipeline.add_pipeline_variable(repeator(keys=keys))
    d1_loopkeys.static_args = repeator(pipeline=pipeline, keys=keys)

    main_pipeline.compile_pipeline([d0_model_load, d1_loopkeys])

    main_pipeline.run()


docker_image = 'us.gcr.io/dl-solutions-dev/nikhil/images/appsci_utils_py_35_git1efc51ff9f3faa62c40971bb190af111f43133b3_substations@sha256:'
docker_image += '0a990ec7e7d56cb56bbb5b6c53e806b456edfdd9160fdd648928aad241ac5578'  # compiled binary aav, ffx, ..

# Flag to run it locally or in tasks
run_local = True
if run_local:
    model_path = '/home/nikhil/Documents/gcloud/nik-dev-gpu/from_data/wind_turbines/deployment/model_pretrain_resnet_512_dice_wbce_cvResNet_tap100_v2.hdf5'
else:
    model_path = '/festivus/nikhil-dev/wind_turbines/saved_models/model_pretrain_resnet_512_dice_wbce_cvResNet_tap100_v2.hdf5'

# Here, we get the keys, and then spin the worker, to which the function `main_func` is submitted.
# Depending on the `run_local` flag, that function will either be run locally or run on Tasks.
a0_getkeys = Component(name='getkeys', func=get_keys)
a0_getkeys.to_record = repeator('keys')

a1_spinworker = Component(name='spinworker', func=spinworker)
a1_spinworker.static_args = repeator(task_function=main_func, run_local=run_local,
                                     docker_image=docker_image, model_path=model_path)
a1_spinworker.runtime_args = repeator(dlkeys=a0_getkeys.FutureVariable.keys)

worker = Pipeline('worker')
worker.compile_pipeline([a0_getkeys, a1_spinworker])
worker.run()

# Summarize
# Pipeline.summarize()
# Pipeline.summarize(visualize_path='wind_turbines.png')
print('\nTOTAL TIME: ', datetime.now() - startTime)
