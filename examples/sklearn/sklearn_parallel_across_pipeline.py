from datetime import datetime
import joblib
import os
import sys

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

sys.path.append('../..')
from deploy import Component, Pipeline, repeator
from local_func_library import get_keys, spinworker


startTime = datetime.now()

num_cores = 3

def main_func(keys, model_path):
    from functools import partial

    import numpy as np

    from appsci_utils.generator.transforms import cast_transform, rescale_feature_transform
    from local_func_library import get_raster, loop_key_deployer_parallel, preprocess


    transforms = [partial(cast_transform, feature_type=np.float32, target_type=np.float32),
                  partial(rescale_feature_transform, shift=128.0, scale=128.0)]

    pipeline = Pipeline('pre_predict_post_upload345')
    pipeline.add_pipeline_variable(repeator(products=['airbus:oneatlas:spot:v1']))
    pipeline.add_pipeline_variable(repeator(break_tilesize=4096, unpadded_tilesize=4096, pad=0))

    main_pipeline = Pipeline('main')

    def load_model_func(model_path):
        with open('dummy_svm_m1.pkl', 'rb') as f:
            model = joblib.load(f)
        return locals()


    def model_pred_wrapper(batch_imgs, break_tilesize, pad, unpadded_tilesize, model):
        batch_imgs = batch_imgs.reshape((batch_imgs.shape[0], batch_imgs.shape[1] * batch_imgs.shape[2], batch_imgs.shape[3]))
        batch_proba = model.predict(batch_imgs[0, ...])
        return locals()


    d0_model_load = Component(name='load_model', func=load_model_func)
    d0_model_load.static_args = repeator(model_path=model_path)
    d0_model_load.to_record=repeator('model')


    c0_raster = Component(name='get_raster', func=get_raster, debug=False)
    c0_raster.static_args = repeator(bands=['red', 'green', 'blue'], start_datetime='2014-01-01', end_datetime='2019-01-01')
    c0_raster.runtime_args = {'products': pipeline.FutureVariable.products}
    c0_raster.to_record = repeator('batch_imgs')

    c1_preprocess = Component(name='preprocessing', func=preprocess, debug=False)
    c1_preprocess.static_args=repeator(transforms=transforms)
    c1_preprocess.runtime_args={'batch_imgs': c0_raster.FutureVariable.batch_imgs}
    c1_preprocess.to_record=repeator('batch_imgs')
    # c1_preprocess.to_print=['batch_imgs']

    c3_model_predict = Component(name='predict', func=model_pred_wrapper)
    c3_model_predict.runtime_args={'batch_imgs': c1_preprocess.FutureVariable.batch_imgs,
                  'break_tilesize': pipeline.FutureVariable.break_tilesize,
                  'unpadded_tilesize': pipeline.FutureVariable.break_tilesize,
                  'model': d0_model_load.FutureVariable.model,
                  'pad': pipeline.FutureVariable.pad}

    pipeline.compile_pipeline([c0_raster, c1_preprocess, c3_model_predict], parallelize=False)


    d1_loopkeys = Component(name='loopkeys', func=loop_key_deployer_parallel)
    main_pipeline.add_pipeline_variable(repeator(keys=keys))
    d1_loopkeys.static_args=repeator(pipeline=pipeline, keys=keys, num_cores=num_cores)

    main_pipeline.compile_pipeline([d0_model_load, d1_loopkeys], parallelize=False)

    main_pipeline.run()

    # main_pipeline.run()


docker_image = 'us.gcr.io/dl-solutions-dev/nikhil/images/appsci_utils_py_35_git1efc51ff9f3faa62c40971bb190af111f43133b3_substations@sha256:'
docker_image += '0a990ec7e7d56cb56bbb5b6c53e806b456edfdd9160fdd648928aad241ac5578'  # compiled binary aav, ffx, ..

run_local = True
if run_local:
    model_path = '/home/nikhil/Documents/gcloud/nik-dev-gpu/from_data/wind_turbines/deployment/model_pretrain_resnet_512_dice_wbce_cvResNet_tap100_v2.hdf5'
else:
    model_path = '/festivus/nikhil-dev/wind_turbines/saved_models/model_pretrain_resnet_512_dice_wbce_cvResNet_tap100_v2.hdf5'

a0_getkeys = Component(name='getkeys', func=get_keys)
a0_getkeys.to_record = repeator('keys')

a1_spinworker = Component(name='spinworker', func=spinworker)
a1_spinworker.static_args=repeator(task_function=main_func, run_local=run_local, docker_image=docker_image, model_path=model_path)
a1_spinworker.runtime_args=repeator(dlkeys=a0_getkeys.FutureVariable.keys)

worker = Pipeline('worker')
worker.compile_pipeline([a0_getkeys, a1_spinworker])
worker.run()

Pipeline.summarize(visualize_path='sklearn.png')
print('TOTAL TIME: ', datetime.now() - startTime)
