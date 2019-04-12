from collections import defaultdict
import six

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import DBSCAN
import tensorflow as tf

from appsci_utils.image_processing.image_tiling import generate_tiles_tight
import descarteslabs as dl


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Helper Functions
# Meant to be used only "within" other Component functions defined here, and not
# on their onw.
# All the functions defined here should follow the guidelines from Contributing
# to the Function Library
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def postprocess_wt(batch_imgs, small_tile, tilesize=None, **kwargs):
    PROB_THRESHOLD = 0.5
    MIN_BLOB_COUNT = 3
    MIN_CLUSTER_COUNT_1 = 4
    MAX_CLUSTER_DISTANCE_1 = 600 / 16
    MIN_CLUSTER_COUNT_2 = 3 * MIN_CLUSTER_COUNT_1
    MAX_CLUSTER_DISTANCE_2 = 3 * MAX_CLUSTER_DISTANCE_1
    MIN_ALL_TP_COUNT = 10

    #if small_tile:
    #else:
    #    batch_output = np.zeros_like(batch_imgs)
    batch_output = np.empty((batch_imgs.shape[0], tilesize, tilesize), dtype='uint8')
    for batch_iter, img in enumerate(batch_imgs):
        if small_tile:
            img = img.squeeze()

            # STEP 1)
            img = np.where(img < PROB_THRESHOLD, 0, 1).astype('uint8')
            img_ht, img_wd = img.shape[:2]

            current_output = img.copy().astype('uint8')
            current_output = Image.fromarray(current_output.squeeze())

            # STEP 2)
            current_output = np.asarray(current_output.resize((16*img_ht, 16*img_wd)))
            img_ht, img_wd = current_output.shape[:2]
            x_start = max(0, int((img_wd - tilesize)/2))
            x_end = min(img_wd, x_start + tilesize)
            y_start = max(0, int((img_ht - tilesize)/2))
            y_end = min(img_ht, y_start + tilesize)

            current_output = current_output[y_start:y_end, x_start:x_end]

            batch_output[batch_iter] = current_output.astype('uint8')
            continue
        try:
            img = img.squeeze()
            img_ht, img_wd = tilesize, tilesize  # img.shape[:2]
            #img = Image.fromarray(img.astype('uint8'))
            # STEP 1)
            #img = img.resize((int(img_ht/16), int(img_wd/16)))
            #img = np.asarray(img, dtype='uint8')
            img = np.where(img < 0.5, 0, 1).astype('uint8')
            current_output = np.zeros_like(img).astype('uint8')
            if np.max(img) == 0:
                current_output.fill(0)
            else:
                # STEP 2)
                img = ndimage.binary_closing(img, structure=np.ones((3,3))).astype('uint8')
                centroids = []
                pts_x = []
                pts_y = []
                # STEP 3)
                label_array, num_features = ndimage.label(img.squeeze())
                for feat in range(1, num_features+1):
                    # Loop over connected components
                    current_xs, current_ys = np.where(label_array == feat)
                    # STEP 4)
                    if len(current_xs) < MIN_BLOB_COUNT:
                        current_output[current_xs, current_ys] = 0  # COLORS[0]  # RED
                    else:
                        centroid_x, centroid_y = np.mean(current_xs), np.mean(current_ys)
                        centroids.append([centroid_x, centroid_y])
                        pts_x.append(current_xs)
                        pts_y.append(current_ys)
                        current_output[current_xs, current_ys] = 1  # COLORS[1]  # GREEN

                # STEP 5)
                if len(centroids) != 0:  # Post-processing using density clustering
                    model_1 = DBSCAN(eps=MAX_CLUSTER_DISTANCE_1, min_samples=1, metric='euclidean', n_jobs=-1)
                    centroids = np.array(centroids)
                    pts = np.column_stack((centroids[:, 0], centroids[:, 1]))
                    clusters = model_1.fit_predict(pts)

                    cluster_dict_count = defaultdict(lambda: 0)  # Record the number of points belonging to a given cluster
                    cluster_dict_indices = defaultdict(lambda: [])
                    # To map the cluster number to all the indices of all centroids that belong to that cluster

                    # Variables starting with `tp1` record results relevant to true positives found in step 5a
                    tp1_centroids = []
                    tp1_counts = []
                    recheck_pts = []
                    recheck_pts_idxs = []
                    tp1_points = []

                    for loop_cluster, cluster in enumerate(clusters):
                        cluster_dict_count[cluster] += 1
                        cluster_dict_indices[cluster].append(loop_cluster)
                    # STEP 5a)
                    for loop_cluster_dict, count in cluster_dict_count.items():
                        if count < MIN_CLUSTER_COUNT_1:
                            # This cluster does not have enough neighbors.
                            # Record it for further processing
                            for idx in cluster_dict_indices[loop_cluster_dict]:
                                current_output[pts_x[idx], pts_y[idx]] = 0  # COLORS[5]  # PEACH ORANGE
                                recheck_pts.append(centroids[idx])
                                recheck_pts_idxs.append(idx)
                        else:
                            # Definite TP cluster
                            # Get the centroid of this cluster
                            cluster_centroid_x = np.mean(np.array([centroids[idx][0] for idx in \
                                                         cluster_dict_indices[loop_cluster_dict]]))
                            cluster_centroid_y = np.mean(np.array([centroids[idx][1] for idx in \
                                                         cluster_dict_indices[loop_cluster_dict]]))
                            tp1_points.append([cluster_centroid_x, cluster_centroid_y])
                            tp1_counts.append(count)

                    # Round 2 of clustering
                    # STEP 5b)
                    num_recheck = len(recheck_pts)
                    num_tp1 = sum(tp1_counts)
                    # Step 5b)
                    if num_tp1 == 0:
                        for idx in range(num_recheck+num_tp1):
                            current_output[pts_x[idx], pts_y[idx]] = 0  # COLORS[3]  # YELLOW
                    elif num_tp1 > MIN_ALL_TP_COUNT:
                        for idx in range(num_recheck):
                            original_idx = recheck_pts_idxs[idx]
                            current_output[pts_x[original_idx], pts_y[original_idx]] = 1  # COLORS[2]  # BLUE
                    else:
                        if num_recheck > 0:
                            num_tp1_clusters = len(tp1_points)
                            num_converted_tp = 0
                            all_points = np.array(recheck_pts + tp1_points)
                            model_2 = DBSCAN(eps=MAX_CLUSTER_DISTANCE_2, min_samples=1, metric='euclidean', n_jobs=-1)
                            clusters = model_2.fit_predict(all_points)

                            # Check for recheck_pts
                            # Record the number of points belonging to a given cluster
                            cluster_dict_count = defaultdict(lambda: 0)

                            for loop_recheck_pts in range(num_recheck):
                                cluster_id = clusters[loop_recheck_pts]
                                cluster_dict_count[cluster_id] += 1
                            for loop_tp1_pts in range(num_tp1_clusters):
                                cluster_id = clusters[loop_tp1_pts]
                                cluster_dict_count[cluster_id] += tp1_counts[loop_tp1_pts]

                            for loop_recheck_pts in range(num_recheck):
                                cluster_id = clusters[loop_recheck_pts]
                                cluster_count = cluster_dict_count[cluster_id]
                                if cluster_count >= MIN_CLUSTER_COUNT_2:
                                    # Near a big group. Relabel as TP
                                    original_idx = recheck_pts_idxs[loop_recheck_pts]
                                    current_output[pts_x[original_idx], pts_y[original_idx]] = 1
                                    # COLORS[6]  # LIGHT BLUE
                                    num_converted_tp += 1
            current_output = Image.fromarray(current_output.squeeze())
            current_output = np.asarray(current_output.resize((img_ht, img_wd)))
            batch_output[batch_iter] = (255*current_output).astype('uint8')
        except Exception as ex:
            print ('Could not do the post-processing')
            raise
    return batch_output


def model_predict_func(model, img):
    pred_func = model.predict(img)
    return pred_func


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Component Functions
# All the functions defined here should follow the guidelines from Contributing
# to the Function Library
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def demo1(inp_image, add_value=0, mul_value=1):
    inp_image += add_value
    inp_image *= mul_value
    return locals()


def demo2(inp_image, threshold_value):
    inp_image[inp_image < threshold_value] = 0
    inp_image[inp_image >= threshold_value] = 1
    dummy_variable = 5
    return locals()


def demo3(inp_image, add_value):
    print('I received the add_value as: ', add_value)
    inp_image += add_value
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


def load_model_func(model_path):
    model = tf.keras.models.load_model(model_path)
    return locals()


def model_pred_wrapper(batch_imgs, model, break_tilesize, pad, unpadded_tilesize):
    batch_proba = None
    batch_size = batch_imgs.shape[0]
    for batch_iter, img in enumerate(batch_imgs):
        for tile, src, dest in generate_tiles_tight(img, tilesize=break_tilesize, pad=pad):
            #p start
            proba = model_predict_func(model, tile[np.newaxis, ...])
            if batch_proba is None:  # Initialize a proba size
                batch_proba = np.empty((batch_size, int(unpadded_tilesize/16), int(unpadded_tilesize/16)) + proba.shape[3:],
                                        dtype='float32')
            # dest contains coords wrt the original image size. However, batch_proba here is initialized to
            # trim out padded regions. Therefore, we need to subtract the pad value from dest slices
            dest = (slice(dest[0].start-pad, dest[0].stop-pad), slice(dest[1].start-pad, dest[1].stop-pad)) \
                   + dest[2:]
            batch_proba[batch_iter][dest] = proba[0]
    return locals()


def postprocess_wrapper(batch_imgs, batch_ids, upload_dtype, break_tilesize, pad, unpadded_tilesize, batch_meta, batch_keys, batch_epsg, get_unpadded_meta_keys, postprocess_func):
    image_ids = []
    upload_kwargs = []
    batch_size = batch_imgs.shape[0]
    unpadded_batch_meta, unpadded_batch_keys, unpadded_tilesize, pad = \
        get_unpadded_meta_keys(batch_meta, batch_keys, batch_epsg)

    batch_proba = None
    for batch_iter, img in enumerate(batch_imgs):
        """
        for tile, src, dest in generate_tiles_tight(img, tilesize=break_tilesize, pad=pad):
            #p start
            proba = postprocess_func(proba, tilesize=unpadded_tilesize, small_tile=True)
            #pipeline.run()
            # p end
            if batch_proba is None:  # Initialize a proba size
                batch_proba = np.empty((batch_size, unpadded_tilesize, unpadded_tilesize) + proba.shape[3:],
                                        dtype=upload_dtype)
            # dest contains coords wrt the original image size. However, batch_proba here is initialized to
            # trim out padded regions. Therefore, we need to subtract the pad value from dest slices
            dest = (slice(dest[0].start-pad, dest[0].stop-pad), slice(dest[1].start-pad, dest[1].stop-pad)) \
                   + dest[2:]
            batch_proba[batch_iter][dest] = proba[0]
            # proba[0] and not just proba, because postprocess_func respects the batch axis
        """
        batch_proba = postprocess_func(batch_imgs, tilesize=unpadded_tilesize,
                                       small_tile=False).astype('uint8')
        image_id = '{}_{}.tif'.format(batch_ids[batch_iter][-1], unpadded_batch_keys[batch_iter])
        image_ids.append(image_id)
        unpadded_meta = unpadded_batch_meta[batch_iter]
        current_upload_kwargs = {'wkt_srs': unpadded_meta['coordinateSystem']['wkt'],
                                 'geotrans': unpadded_meta['geoTransform'],
                                 'raster_meta': unpadded_meta}
        upload_kwargs.append(current_upload_kwargs)
    return locals()


def get_raster(key, products, bands, start_datetime, end_datetime, get_ids_func_extra_args=None, alpha_masking=True):
    #print('--------------------get raster calleddd-------------')
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
    """
    img = np.ndarray((1024, 1024, 3), dtype='uint8'); meta = {'e': 'haha'}
    meta = {'bands': [{'mask': {'overviews': [], 'flags': ['PER_DATASET', 'ALPHA']}, 'description': {'resolution_unit': 'm', 'product': 'airbus:oneatlas:spot:v1', 'vendor_order': 1, 'dtype': 'Byte', 'name_vendor': 'Red', 'id': 'airbus:oneatlas:spot:v1:red', 'type': 'spectral', 'default_range': [0, 255], 'nbits': 8, 'owner_type': 'core', 'name': 'red', 'name_common': 'red', 'color': 'Red', 'data_range': [0, 255], 'resolution': 1.5}, 'metadata': {'': {'NBITS': '8'}}, 'block': [1024, 1], 'colorInterpretation': 'Red', 'type': 'Byte', 'band': 1}, {'mask': {'overviews': [], 'flags': ['PER_DATASET', 'ALPHA']}, 'description': {'resolution_unit': 'm', 'product': 'airbus:oneatlas:spot:v1', 'vendor_order': 2, 'dtype': 'Byte', 'name_vendor': 'Green', 'id': 'airbus:oneatlas:spot:v1:green', 'type': 'spectral', 'default_range': [0, 255], 'nbits': 8, 'owner_type': 'core', 'name': 'green', 'name_common': 'green', 'color': 'Green', 'data_range': [0, 255], 'resolution': 1.5}, 'metadata': {'': {'NBITS': '8'}}, 'block': [1024, 1], 'colorInterpretation': 'Green', 'type': 'Byte', 'band': 2}, {'mask': {'overviews': [], 'flags': ['PER_DATASET', 'ALPHA']}, 'description': {'resolution_unit': 'm', 'product': 'airbus:oneatlas:spot:v1', 'vendor_order': 3, 'dtype': 'Byte', 'name_vendor': 'Blue', 'id': 'airbus:oneatlas:spot:v1:blue', 'type': 'spectral', 'default_range': [0, 255], 'nbits': 8, 'owner_type': 'core', 'name': 'blue', 'name_common': 'blue', 'color': 'Blue', 'data_range': [0, 255], 'resolution': 1.5}, 'metadata': {'': {'NBITS': '8'}}, 'block': [1024, 1], 'colorInterpretation': 'Blue', 'type': 'Byte', 'band': 3}, {'metadata': {'': {'NBITS': '1'}}, 'block': [1024, 1], 'colorInterpretation': 'Alpha', 'type': 'Byte', 'band': 4, 'description': {'physical_range': [0, 1], 'resolution_unit': 'm', 'data_unit_description': '0: nodata; 1: valid data', 'product': 'airbus:oneatlas:spot:v1', 'dtype': 'Byte', 'data_unit': 'unitless', 'default_range': [0, 1], 'nbits': 1, 'owner_type': 'core', 'name': 'alpha', 'id': 'airbus:oneatlas:spot:v1:alpha', 'resolution': 1.5, 'type': 'mask', 'data_range': [0, 1], 'color': 'Alpha'}}], 'files': [], 'metadata': {'': {'Corder': 'RPCL', 'id': 'airbus:oneatlas:spot:v1:SPOT7_201801161653326_2660025101_R3C1'}}, 'geoTransform': [578336.0, 1.5, 0.0, 3667968.0, 0.0, -1.5], 'wgs84Extent': {'coordinates': [[[-98.1600374, 33.1476354], [-98.1601694, 33.1337819], [-98.1437037, 33.1336698], [-98.1435692, 33.1475232], [-98.1600374, 33.1476354]]], 'type': 'Polygon'}, 'cornerCoordinates': {'center': [579104.0, 3667200.0], 'lowerRight': [579872.0, 3666432.0], 'lowerLeft': [578336.0, 3666432.0], 'upperRight': [579872.0, 3667968.0], 'upperLeft': [578336.0, 3667968.0]}, 'driverShortName': 'MEM', 'driverLongName': 'In Memory Raster', 'size': [1024, 1024], 'coordinateSystem': {'wkt': 'PROJCS["WGS 84 / UTM zone 14N",\n    GEOGCS["WGS 84",\n        DATUM["WGS_1984",\n            SPHEROID["WGS 84",6378137,298.257223563,\n                AUTHORITY["EPSG","7030"]],\n            AUTHORITY["EPSG","6326"]],\n        PRIMEM["Greenwich",0,\n            AUTHORITY["EPSG","8901"]],\n        UNIT["degree",0.0174532925199433,\n            AUTHORITY["EPSG","9122"]],\n        AUTHORITY["EPSG","4326"]],\n    PROJECTION["Transverse_Mercator"],\n    PARAMETER["latitude_of_origin",0],\n    PARAMETER["central_meridian",-99],\n    PARAMETER["scale_factor",0.9996],\n    PARAMETER["false_easting",500000],\n    PARAMETER["false_northing",0],\n    UNIT["metre",1,\n        AUTHORITY["EPSG","9001"]],\n    AXIS["Easting",EAST],\n    AXIS["Northing",NORTH],\n    AUTHORITY["EPSG","32614"]]'}}

    current_epsg = '4312'  # dltile['properties']['cs_code']
    ids = ['3234:145', '3240459:3490']
    """
    current_epsg = dltile['properties']['cs_code']
    batch_imgs.append(img)
    batch_keys.append(key)
    batch_ids.append(ids)
    batch_meta.append(meta)
    batch_epsg.append(current_epsg)
    batch_imgs = np.array(batch_imgs)
    import time
    if '14:51' in key:
        batch_imgs.fill(0)
    elif '14:50' in key:
        batch_imgs.fill(10)
    elif '13:50' in key:
        batch_imgs.fill(3)
    else:
        raise
    # time.sleep(3)
    #print('--------------------get raster returning-------------', batch_imgs.shape)
    return locals()


def uploader(batch_proba, image_ids, upload_kwargs, products):
    #print(image_ids)
    #print(upload_kwargs)
    #print(products)
    return locals()


def get_keys():
    keys = ['1024:0:1.5:14:51:2387', '1024:0:1.5:14:50:2387']
    # keys = [keys[0]]
    keys = ['1024:0:1.5:14:51:2387', '1024:0:1.5:14:50:2387', '1024:0:1.5:14:49:2387', '1024:0:1.5:14:48:2387', '1024:0:1.5:14:52:2387']
    keys = ['1024:0:1.5:13:51:2387', '1024:0:1.5:13:50:2387', '1024:0:1.5:13:49:2387', '1024:0:1.5:13:48:2387', '1024:0:1.5:13:52:2387']
    keys = ['1024:0:1.5:12:51:2387', '1024:0:1.5:12:50:2387', '1024:0:1.5:12:49:2387', '1024:0:1.5:12:48:2387', '1024:0:1.5:12:52:2387']
    keys = ['1024:0:1.5:15:51:2387', '1024:0:1.5:15:50:2387', '1024:0:1.5:15:49:2387', '1024:0:1.5:15:48:2387', '1024:0:1.5:15:52:2387']
    keys *= 20
    keys = keys[:3]
    keys = ['1024:0:1.5:14:51:2387', '1024:0:1.5:14:50:2387', '1024:0:1.5:13:50:2387']
    return locals()


def loop_key_deployer(keys, pipeline):
    for i in keys:
        pipeline.run(kwargs=dict(key=i))
    pipeline.wait()
    return locals()


def loop_key_deployer_pipeline_parallel(keys, pipeline, predict_pipeline):
    my_dict = {'key': i for i in keys}
    # pipeline.map(kwargs=my_dict)
    for i in keys:
        pipeline.run(key=i)
    #for _ in keys:
    #    predict_pipeline.run()
    return locals()


from multiprocessing import Pool, Process
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
