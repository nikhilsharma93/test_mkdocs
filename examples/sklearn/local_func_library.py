"""Define some dummy functions locally. Alternatively, you could import them from the function library."""

import time

import numpy as np
import tensorflow as tf

import descarteslabs as dl
from appsci_utils.image_processing.image_tiling import generate_tiles_tight


def get_keys():
    keys = ['1024:0:1.5:14:51:2387', '1024:0:1.5:14:50:2387', '1024:0:1.5:13:50:2387']
    return locals()


def preprocess(batch_imgs, transforms):
    """ Preprocess the input by applying the required transforms.

    Parameters
    ----------
    batch_imgs: ndarray
        Input batch of images

    Returns
    -------
    batch_imgs: ndarray
        Output ndarray

    """
    batch_imgs = batch_imgs.astype('float32')
    for batch_iter, img in enumerate(batch_imgs):
        for transform in transforms:
            img, _ = transform(img)
        batch_imgs[batch_iter] = img
    return locals()


def get_raster(key, products, bands, start_datetime, end_datetime, get_ids_func_extra_args=None, alpha_masking=True):
    batch_imgs = list()
    batch_keys = list()
    batch_ids = list()
    batch_meta = list()
    batch_epsg = list()

    available_bands = dl.metadata.bands(products=products)
    alpha_present = [True for i in available_bands if 'alpha' == i['name']]
    product_has_alpha = True if any(alpha_present) else False

    raster_client = dl.Raster()
    metadata_client = dl.Metadata()

    try:
        dltile = raster_client.dltile(key)
    except:
        raster_client = dl.Raster()
        dltile = raster_client.dltile(key)
    get_ids_func_extra_args = get_ids_func_extra_args or {}
    ids = metadata_client.ids(products=products,
                          start_datetime=start_datetime,
                          end_datetime=end_datetime,
                          dltile=dltile,
                          **get_ids_func_extra_args)
    if len(ids) != 0:
        if alpha_masking:
            if 'alpha' in bands:
                # Could add a check here if alpha exists, since we already have the bool `product_has_alpha`
                # But it looks better to not do that and let the process fail, so that the user is aware of it.
                warnings.simplefilter('default', UserWarning)
                warnings.warn('\nGot `alpha` in bands and alpha_masking is also set as True. \n \
                               If you provided alpha just to implement mosaicing, \
                               re-run this without the alpha band.', UserWarning)
                img, meta = raster_client.ndarray(ids, dltile=dltile, bands=bands, **get_ndarray_func_extra_args)
            else:
                if product_has_alpha:
                    img, meta = raster_client.ndarray(ids, dltile=dltile, bands=bands+['alpha'],
                                                 )
                    img = img[..., :-1]
                else:
                    img, meta = raster_client.ndarray(ids, dltile=dltile, bands=bands,
                                                 )
        else:
            img, meta = raster_client.ndarray(ids, dltile=dltile, bands=bands,)
    current_epsg = dltile['properties']['cs_code']
    batch_imgs.append(img)
    batch_keys.append(key)
    batch_ids.append(ids)
    batch_meta.append(meta)
    batch_epsg.append(current_epsg)
    batch_imgs = np.array(batch_imgs)
    return locals()


def loop_key_deployer_parallel(keys, pipeline, num_cores):
    pipeline.map(kwargs=dict(key=keys), num_cores=num_cores)


def spinworker(dlkeys,
               task_function,
               model_path,
               n_tiles=None,
               run_local=True,
               n_per_task=1,
               job_name=None,
               docker_image=None,
               cpu_memory='6Gi',
               num_cores=2,
               async_create_function_optional_kwargs=None):
    n_tiles_in_ROI = len(dlkeys)

    # If requested, limit number of tile
    if n_tiles:
        n_tiles = min(n_tiles, len(dlkeys))
        dlkeys = dlkeys[0:n_tiles]

    if run_local:
        print('\nRunning locally...')
        print('Processing {} tiles\n\n'.format(len(dlkeys)))
        task_function(dlkeys, model_path)

    else:
        # Dlrun setup
        if job_name is None:
            job_name = 'test_run'
        # For some reason, at the time this was written, the job name was required to be of 63 characters at max
        if len(job_name) > 63:
            warnings.simplefilter('default', UserWarning)
            warnings.warn('\n\nTrimming job name to the first 63 characters as per requirement', UserWarning)
            job_name = job_name[:63]

        # Split tiles into chunks
        dltile_chunks = [dlkeys[i:i + n_per_task] for i in six.moves.range(0, len(dlkeys), n_per_task)]
        print('Found {} tiles in ROI. Process {} tiles in {} chunks'.format(
            n_tiles_in_ROI, len(dlkeys), len(dltile_chunks)))

        total_chunks = len(dltile_chunks)

        # -------------------
        # Define the actual tasks.
        # -------------------
        async_task = dl.Tasks()

        # First time user should execute the following line
        # async_task.create_namespace()

        create_function_kwargs = dict(name=job_name,
                                      cpus=num_cores,
                                      image=docker_image,
                                      memory=cpu_memory,
                                      include_modules=['deploy', 'func_library', 'exceptions'])
        if async_create_function_optional_kwargs is not None:
            create_function_kwargs.update(async_create_function_optional_kwargs)

        async_deploy_dltiles = async_task.create_function(task_function, **create_function_kwargs)

        print('\nFunction created. Now deploying using tasks...')

        tasks = []
        for counter, chunk in enumerate(dltile_chunks):
            print('Deploying chunk {} of {}'.format(counter+1, total_chunks))
            task = async_deploy_dltiles(chunk, model_path=model_path)
            #task['dlkeys'] = chunk  # Add dlkeys names for later reference
            #tasks.append(task)
        n_tasks = len(tasks)
        print('\n{} tasks'.format(n_tasks))
    return locals()
