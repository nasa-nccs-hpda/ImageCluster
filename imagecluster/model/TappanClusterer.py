
import pathlib
import logging
import os
import tqdm

import numpy as np
from osgeo import gdal
from sklearn import cluster

from imagecluster.model.Glcm import GLCM
from imagecluster.model.utils import TqdmLoggingHandler, getBaseCmd
from core.model.SystemCommand import SystemCommand
from core.model.GeospatialImageFile import GeospatialImageFile


# -----------------------------------------------------------------------------
# Class TappanClusterer
# -----------------------------------------------------------------------------
class TappanClusterer(object):

    GLCM_DISTANCES: list = [1]
    GLCM_FEATURES: list = ["homogeneity", "mean"]
    GLCM_GRID_SIZE: int = 30
    GLCM_BIN_SIZE: int = 8
    GLCM_ANGLES: list = [0]

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config) -> None:
        self.conf = config

    # -------------------------------------------------------------------------
    # setup_logging
    # -------------------------------------------------------------------------
    def setup_logging(self, log_file: str) -> logging.Logger:
        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Create a file handler for the log file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create a stream handler for stdout
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(logging.DEBUG)

        # Define the log format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    # -------------------------------------------------------------------------
    # reshape
    # -------------------------------------------------------------------------
    def reshape(self,
                img: np.ndarray,
                logger: logging.Logger) -> np.ndarray:

        logger.debug(
            f'Img shape before transpose: {img.shape}')

        channel_last_array = np.transpose(img, (1, 2, 0))

        output_shape = channel_last_array[:, :, 0].shape

        logger.debug(
            f'Clipped shape after transpose: {channel_last_array.shape}')

        img1d_channels = channel_last_array.reshape(-1,
                                                    channel_last_array.shape[-1])

        logger.debug(f'Clipped shape after 1d trf: {img1d_channels.shape}')

        return img1d_channels, output_shape

    # -------------------------------------------------------------------------
    # read_img
    # -------------------------------------------------------------------------
    def read_img(self,
                 clipped_path: str,
                 logger: logging.Logger) -> np.ndarray:

        logger.info(f'Reading in {clipped_path}')

        clipped_dataset = gdal.Open(clipped_path)

        if clipped_dataset is None:

            msg = f'Could not open {clipped_dataset} with GDAL'

            logger.error(msg)

            raise RuntimeError(msg)

        clipped_dataset_array = clipped_dataset.ReadAsArray()

        return clipped_dataset_array

    # -------------------------------------------------------------------------
    # agglomerative
    # -------------------------------------------------------------------------
    def agglomerative(self,
                      clipped_path: str,
                      logger: logging.Logger) -> np.ndarray:

        img_channels = self.read_img(clipped_path, logger)

        logger.info(img_channels.shape)

        img_channels = self.calculate_texture_channels(img_channels, logger)

        logger.info(img_channels.shape)

        img1d_channels, output_shape = self.reshape(img_channels, logger)

        logger.info(img1d_channels.shape)

        params = {
            'n_clusters': self.conf['num_clusters'],
        }

        cl_aglom = cluster.AgglomerativeClustering(**params)

        model = cl_aglom.fit(img1d_channels)

        img_cl = model.labels_

        img_cl = img_cl.reshape(output_shape)

        return img_cl

    # -------------------------------------------------------------------------
    # write_out_clustered_img
    # -------------------------------------------------------------------------
    def write_out_clustered_img(self,
                                cluster_image: np.ndarray,
                                clipped_image_path: str,
                                algorithm: str,
                                output_dir: str,
                                nclusters: int,
                                logger: logging.Logger) -> int:

        clipped_image_dataset = GeospatialImageFile(
            clipped_image_path).getDataset()

        clipped_image_name = os.path.basename(clipped_image_path)

        logger.debug(f'Cluster image shape: {cluster_image.shape}')

        post_str = f'{algorithm}{nclusters}.tif'

        output_name = clipped_image_name.replace('data.tif', post_str)

        output_path = os.path.join(output_dir, output_name)

        logger.info(f'Writing to {output_path}')

        output_srs = clipped_image_dataset.GetProjection()

        output_trf = clipped_image_dataset.GetGeoTransform()

        driver = gdal.GetDriverByName('GTiff')

        dst_ds = driver.Create(output_path,
                               cluster_image.shape[1],
                               cluster_image.shape[0],
                               1,
                               gdal.GDT_Byte)

        dst_ds.SetProjection(output_srs)

        dst_ds.SetGeoTransform(output_trf)

        dst_band = dst_ds.GetRasterBand(1)

        dst_band.WriteArray(cluster_image.astype(np.uint8))

        dst_band.FlushCache()

        dst_ds.FlushCache()

        return 1

    # -------------------------------------------------------------------------
    # calculate_texture_channels
    # -------------------------------------------------------------------------
    def calculate_texture_channels(self,
                                   img_channels: np.ndarray,
                                   logger=logging.Logger) -> np.ndarray:

        glcm = GLCM(features=self.GLCM_FEATURES,
                    distances=self.GLCM_DISTANCES,
                    grid_size=self.GLCM_GRID_SIZE,
                    bin_size=self.GLCM_BIN_SIZE,
                    angles=self.GLCM_ANGLES)

        logger.info(glcm.features)

        img_band_one = img_channels[0, :, :]

        texture_channels = glcm.compute_band_features(img_band_one)

        logger.info(len(texture_channels))

        homogeneity_texture_channel = texture_channels[0]

        mean_texture_channel = texture_channels[1]

        homogeneity_texture_channel = np.expand_dims(
            homogeneity_texture_channel, axis=0)

        mean_texture_channel = np.expand_dims(mean_texture_channel, axis=0)

        img_channels = np.append(
            img_channels, homogeneity_texture_channel, axis=0)

        img_channels = np.append(img_channels, mean_texture_channel, axis=0)

        return img_channels

    # -------------------------------------------------------------------------
    # kmeans
    # -------------------------------------------------------------------------
    def kmeans(self,
               clipped_path: str,
               logger: logging.Logger) -> np.ndarray:

        img_channels = self.read_img(clipped_path, logger)

        logger.info(img_channels.shape)

        img_channels = self.calculate_texture_channels(img_channels, logger)

        logger.info(img_channels.shape)

        img1d_channels, output_shape = self.reshape(img_channels, logger)

        logger.info(img1d_channels.shape)

        params = {
            'n_clusters': self.conf['num_clusters'],
            'random_state': self.conf['random_state'],
            'batch_size': self.conf['batch_size'],
        }

        cl_kmeans = cluster.MiniBatchKMeans(**params)

        model = cl_kmeans.fit(img1d_channels)

        img_cl = model.labels_

        img_cl = img_cl.reshape(output_shape)

        return img_cl

    # -------------------------------------------------------------------------
    # clip_geotiff
    # -------------------------------------------------------------------------
    def clip_geotiff(
            self,
            input_path: str,
            output_path: str,
            upper_left_x: float,
            upper_left_y: float,
            window_size_x: int,
            window_size_y: int,
            logger: logging.Logger) -> int:

        dataset = gdal.Open(input_path)

        dataset_geotransform = dataset.GetGeoTransform()

        ulx = upper_left_x

        uly = upper_left_y

        x_scale = dataset_geotransform[1]

        y_scale = dataset_geotransform[5]

        lrx = ulx + window_size_x * x_scale

        lry = uly + window_size_y * y_scale

        cmd = getBaseCmd(dataset)

        cmd += (' -te' +
                ' ' + str(ulx) +
                ' ' + str(lry) +
                ' ' + str(lrx) +
                ' ' + str(uly) +
                ' -te_srs' +
                ' "' + dataset.GetSpatialRef().ExportToProj4() +
                '"')

        cmd += ' ' + input_path + ' ' + output_path

        SystemCommand(cmd, logger, True)

        return 1

    # -------------------------------------------------------------------------
    # get_output_name
    # -------------------------------------------------------------------------
    def get_output_name(self,
                        input_path: str,
                        input_identifier: str,
                        square_number: int,
                        output_pre_str: str,
                        output_dir: str) -> str:

        filename = os.path.basename(input_path)

        filename_id = filename.replace(input_identifier, '')

        output_filename = f"{output_pre_str}{square_number}_" + \
            f"{filename_id}_ARD.data.tif"

        output_filepath = os.path.join(output_dir, output_filename)

        return output_filepath

    # -------------------------------------------------------------------------
    # run_one_clip
    # -------------------------------------------------------------------------
    def run_one_clip(self,
                     input_path: str,
                     logger: logging.Logger) -> str:

        output_path = self.get_output_name(
            input_path,
            self.conf['input_identifier'],
            self.conf['square_number'],
            self.conf['output_pre_str'],
            self.conf['output_dir'])

        if not os.path.exists(output_path):

            logger.info(f'{output_path} does not exist. Making.')

            self.clip_geotiff(input_path,
                              output_path,
                              self.conf['upper_left_x'],
                              self.conf['upper_left_y'],
                              self.conf['window_size_x'],
                              self.conf['window_size_y'],
                              logger=logger)

        else:

            logger.info(f'{output_path} already exists')

        return output_path

    # -------------------------------------------------------------------------
    # open_readlines_file
    # -------------------------------------------------------------------------
    def open_readlines_file(self,
                            file_path: str,
                            logger: logging.Logger) -> list:

        if not os.path.exists(file_path):

            msg = f'No file: {file_path}'

            logger.error(msg)

            raise FileNotFoundError(msg)

        with open(file_path, 'r') as file_handler:

            files_to_process = file_handler.readlines()

            files_to_process = [file_to_process.strip()
                                for file_to_process in files_to_process]

            logger.debug(files_to_process)

        return files_to_process

    # -------------------------------------------------------------------------
    # find_file_name
    # -------------------------------------------------------------------------
    def find_file_name(self,
                       input_dir: pathlib.Path,
                       name_pre_str: str,
                       input_identifier: str,
                       logger: logging.Logger) -> pathlib.Path:
        """_summary_

        Args:
            input_dir (pathlib.Path): _description_
            name_pre_str (str): _description_
            logger (logging.Logger): _description_

        Returns:
            pathlib.Path: _description_
        """

        name_regex = f'{name_pre_str}*{input_identifier}'

        matching_paths = list(input_dir.glob(name_regex))

        logger.debug(matching_paths)

        if len(matching_paths) == 0:

            error_msg = 'Could not find file matching pattern' + \
                f' {name_regex} in dir {input_dir}'

            logger.error(error_msg)

            raise FileNotFoundError(error_msg)

        first_file_matching_pattern = str(matching_paths[0])

        logger.debug(first_file_matching_pattern)

        return first_file_matching_pattern

    # -------------------------------------------------------------------------
    # run
    # -------------------------------------------------------------------------
    def run(self):

        square_number = self.conf['square_number']
        output_dir = self.conf['output_dir']

        input_dir = pathlib.Path(self.conf['input_dir'])

        input_identifier = self.conf['input_identifier']

        log_file_name = f'clipCluster{square_number}.log'
        log_file_name = os.path.join(output_dir, log_file_name)
        logger = self.setup_logging(log_file_name)
        logger.debug(f'self.configuration: {self.conf}')

        input_txt_file = self.conf['input_txt_file']

        files_to_clip = self.open_readlines_file(input_txt_file, logger)

        logger.debug(f'Clipping {len(files_to_clip)} files')

        for file_to_clip in tqdm.tqdm(files_to_clip):

            file_to_clip = file_to_clip.replace('M1BS', 'MS')

            file_path_to_clip = self.find_file_name(
                input_dir, file_to_clip,
                input_identifier, logger)

            if self.conf['clip']:

                logger.info('Clipping')

                output_path = self.run_one_clip(
                    file_path_to_clip, logger)

            else:

                logger.info('Not clipping')

                output_path = file_path_to_clip

            if self.conf['clustering']:

                nclusters = self.conf['num_clusters']
                algorithm = self.conf['algorithm']

                logger.info(f'Clustering {output_path} using {algorithm}')

                image_clustered = self.kmeans(output_path, logger)

                self.write_out_clustered_img(
                    image_clustered,
                    output_path,
                    algorithm,
                    output_dir,
                    nclusters,
                    logger)

        return 1
