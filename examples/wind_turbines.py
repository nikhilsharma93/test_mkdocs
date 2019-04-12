import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from deploy import Component, Pipeline, placeholder


def main_func(keys, model_path):
    from functools import partial

    import numpy as np

    from appsci_utils.deployment.deploy import get_unpadded_meta_keys
    from appsci_utils.generator.transforms import cast_transform, rescale_feature_transform
    from deploy import Component, Pipeline, placeholder
    from func_library import postprocess_wt


    transforms = [partial(cast_transform, feature_type=np.float32, target_type=np.float32),
                  partial(rescale_feature_transform, shift=128.0, scale=128.0)]

    pipeline = Pipeline('pre_predict_post_upload')
    pipeline.add_pipeline_variable(placeholder(products=['airbus:oneatlas:spot:v1']))
    pipeline.add_pipeline_variable(placeholder(break_tilesize=1024, unpadded_tilesize=1024, pad=0))

    # model_path = '/home/nikhil/Documents/gcloud/nik-dev-gpu/from_data/wind_turbines/deployment/model_pretrain_resnet_512_dice_wbce_cvResNet_tap100_v2.hdf5'
    """
    print('looading model')
    loaded_model = tf.keras.models.load_model(model_path)
    print('loaded model')
    """

    main_pipeline = Pipeline('main')

    d0_model_load = Component(name='load_model')
    d0_model_load.load_func('load_model_func')
    d0_model_load.static_args = placeholder(model_path=model_path)
    d0_model_load.to_record=placeholder('model')


    c0_raster = Component(name='get_raster')
    c0_raster.load_func('get_raster')
    c0_raster.static_args = placeholder(bands=['red', 'green', 'blue'], start_datetime='2014-01-01', end_datetime='2019-01-01')
    c0_raster.runtime_args = {'products': pipeline.products}
    c0_raster.to_record = placeholder('batch_imgs', 'batch_keys', 'batch_ids', 'batch_meta', 'batch_epsg')

    #from func_library import preprocess
    #my_preprocess = partial(preprocess, transforms=transforms)
    c1_preprocess = Component(name='preprocessing')
    c1_preprocess.load_func('preprocess')
    c1_preprocess.static_args=placeholder(transforms=transforms)
    c1_preprocess.runtime_args={'batch_imgs': c0_raster.batch_imgs}
    c1_preprocess.to_record=placeholder('batch_imgs')
    # c1_preprocess.to_print=['batch_imgs']

    c3_model_predict = Component(name='predict')
    c3_model_predict.load_func('model_pred_wrapper')
    c3_model_predict.runtime_args={'batch_imgs': c1_preprocess.batch_imgs,
                  'break_tilesize': pipeline.break_tilesize,
                  'unpadded_tilesize': pipeline.break_tilesize,
                  'model': d0_model_load.model,
                  'pad': pipeline.pad}
    c3_model_predict.to_record=placeholder('batch_proba')
    c3_model_predict.to_print = ['batch_proba']

    c4_postprocess = Component(name='postprocess')
    c4_postprocess.load_func('postprocess_wrapper')
    c4_postprocess.static_args=placeholder(upload_dtype='uint8', get_unpadded_meta_keys=get_unpadded_meta_keys, postprocess_func=postprocess_wt)
    c4_postprocess.runtime_args={'batch_ids': c0_raster.batch_ids,
                'break_tilesize': pipeline.break_tilesize,
                'unpadded_tilesize': pipeline.unpadded_tilesize,
                'pad': pipeline.pad,
                'batch_meta': c0_raster.batch_meta,
                'batch_epsg': c0_raster.batch_epsg,
                'batch_keys': c0_raster.batch_keys,
                'batch_imgs': c3_model_predict.batch_proba}
    c4_postprocess.to_record=placeholder('batch_proba', 'image_ids', 'upload_kwargs')

    c5_uploader = Component(name='upload')
    c5_uploader.load_func('uploader')
    c5_uploader.runtime_args={'batch_proba': c4_postprocess.batch_proba,
                'image_ids': c4_postprocess.image_ids,
                'upload_kwargs': c4_postprocess.upload_kwargs,
                'products': pipeline.products}
    c5_uploader.to_save={'batch_proba': 'artifacts/proba.png'}

    pipeline.compile_pipeline([c0_raster, c1_preprocess, c3_model_predict, c4_postprocess, c5_uploader])


    d1_loopkeys = Component(name='loopkeys')
    main_pipeline.add_pipeline_variable(placeholder(keys=keys))
    d1_loopkeys.load_func('loop_key_deployer')
    d1_loopkeys.static_args=placeholder(pipeline=pipeline, keys=keys)

    main_pipeline.compile_pipeline([d0_model_load, d1_loopkeys])
                                   #share_from=[ [pipeline, placeholder('N')] ],
                                   #share_with=[ [pipeline, placeholder('model', 'keys')] ])
    # main_pipeline.compile_pipeline([d0_model_load])
    main_pipeline.run()



# main_func(keys=['1024:0:1.5:14:51:2387'], model_path=None)
# """
docker_image = 'us.gcr.io/dl-solutions-dev/nikhil/images/appsci_utils_py_35_git1efc51ff9f3faa62c40971bb190af111f43133b3_substations@sha256:'
docker_image += '0a990ec7e7d56cb56bbb5b6c53e806b456edfdd9160fdd648928aad241ac5578'  # compiled binary aav, ffx, ..

run_local = False
if run_local:
    model_path = '/home/nikhil/Documents/gcloud/nik-dev-gpu/from_data/wind_turbines/deployment/model_pretrain_resnet_512_dice_wbce_cvResNet_tap100_v2.hdf5'
else:
    model_path = '/festivus/nikhil-dev/wind_turbines/saved_models/model_pretrain_resnet_512_dice_wbce_cvResNet_tap100_v2.hdf5'

a0_getkeys = Component(name='getkeys')
a0_getkeys.load_func('get_keys')
a0_getkeys.to_record = placeholder('keys')

a1_spinworker = Component(name='spinworker')
a1_spinworker.load_func('spinworker')
a1_spinworker.static_args=placeholder(task_function=main_func, run_local=run_local, docker_image=docker_image, model_path=model_path)
a1_spinworker.runtime_args=placeholder(dlkeys=a0_getkeys.keys)

worker = Pipeline('worker')
worker.compile_pipeline([a0_getkeys, a1_spinworker])
worker.run()
# """
