# ImageCluster
Using k-means to cluster satellite images 


## Using TappanCluster

### Cloning the code and necessary dependencies
```bash
git clone -b kmeans-cli git@github.com:nasa-nccs-hpda/ImageCluster.git
git clone git@github.com:nasa-nccs-hpda/core.git
```

### Setting up the config and file files
TappanCluster needs two files for running. The first is a text file where each line contains the first portion of the WV filename you are running this application with. 

#### Example
An example of one of these text files is located in `ImageCluster/examples/Tappan01_WV-MS.txt`
```
WV02_20110430_M1BS_103001000A27E100
WV02_20121014_M1BS_103001001B793900
WV02_20130414_M1BS_103001001F227000
WV02_20161026_M1BS_103001005E913900
WV02_20161026_M1BS_103001005FAB0500
WV02_20170107_M1BS_1030010064421600
WV02_20180218_M1BS_1030010078354800
WV02_20180218_M1BS_10300100791C4900
WV02_20181217_M1BS_1030010089CC6D00
WV02_20190413_M1BS_103001008F9EA400
WV02_20190413_M1BS_1030010090262400
```

The second necessary file is the config file, this file sets up the configurations for how the application should behave with the given data. 

#### Example
An example of a config file is located in `ImageCluster/examples/configs/tappan_01.yaml`

```yaml
square_number: <identifying number>

algorithm: kmeans

input_dir: '/path/to/input/dir/'
input_txt_file: 'ImageCluster/examples/Tappan01_WV-MS.txt' # File containing names of the images we want to process
input_identifier: '-ard.tif' # Post-string and file-type of the input files

clip: True # True: clip the image to an extent, False: run kmeans on the input image without changes
upper_left_x: 597415 # Upper-left X in meters of the corner to make clip from
upper_left_y: 1468280 # Upper-left Y in meters of the corner to make clip from
window_size_x: 5000 # Extent (in pixels) of the x axis clip
window_size_y: 5000 # Extent (in pixels) of the y axis clip

clustering: True # True: run clustering, False: no clustering
num_clusters: 60 # Number of clusters to set k to in k-means
random_state: 42 
batch_size: 32

output_pre_str: 'Test' # Pre-string of the output results
# Example if the input image is 'input.tif' the output will be 'Test_input.tif'

output_dir: '/path/to/output/dir'

```

### Running tappan cluster

Currently the ilab-base container does not have all the packages we need, due to that we will need to use conda while the application container is built.

```bash
$ module load anaconda
$ conda activate ilab-pytorch
$ ls
core ImageCluster
(ilab-pytorch) $ export PYTHONPATH=$PWD:$PWD/core:$PWD/ImageCluster
(ilab-pytorch) $ python ImageCluster/imagecluster/view/tappanClusterCLV.py -c <path to config file>
```


#### Example
```bash
$ module load anaconda
$ conda activate ilab-pytorch
$ ls
core ImageCluster
(ilab-pytorch) $ export PYTHONPATH=$PWD:$PWD/core:$PWD/ImageCluster
(ilab-pytorch) $ python ImageCluster/imagecluster/view/tappanClusterCLV.py -c ImageCluster/examples/configs/tappan_01.yaml
```
