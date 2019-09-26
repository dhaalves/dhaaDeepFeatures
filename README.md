

## Usage

#### Dataset Folder Structure
For the images dataset it is expected a folder containing class-named subfolders, each full of images for each label. 

The example folder flower_photos should have a structure like this:
```
~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg
```
#### Example Dataset (flower_photos)
The subfolder names define what label is applied to each image, but the filenames themselves don't matter. 
Example:
 ```bash
 wget http://download.tensorflow.org/example_images/flower_photos.tgz
 tar xzf flower_photos.tgz
 ```

#### Running
Once your images are prepared, and you have pip-installed tensorflow-hub and
a sufficiently recent version of tensorflow, you can run extract deep features with a
command like this:
```bash
python deep_features.py --dataset_folder ~/flower_photos/ --network='inception_v3'
```

## Parameters
```bash
  -dataset_folder       Folder containing image dataset
  -network              default: 'inception_v3'; 
                        Networks avaliable:
                        ['mobilenet_v1_025_224', 'mobilenet_v1_050_224',
                        'mobilenet_v1_075_224', 'mobilenet_v1_100_224',
                        'mobilenet_v2_035_224', 'mobilenet_v2_050_224',
                        'mobilenet_v2_075_224', 'mobilenet_v2_100_224',
                        'mobilenet_v2_130_224', 'mobilenet_v2_140_224',
                        'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152',
                        'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152',
                        'inception_v1', 'inception_v2', 'inception_v3',
                        'inception_resnet_v2', 'nasnet_mobile',
                        'nasnet_large', 'pnasnet_large',
                        'amoebanet_a_n18_f448', 'efficientnet_b0',
                        'efficientnet_b1', 'efficientnet_b2',
                        'efficientnet_b3', 'efficientnet_b4',
                        'efficientnet_b5', 'efficientnet_b6',
                        'efficientnet_b7']
  -batch_size BATCH_SIZE
                        default: 16

```
