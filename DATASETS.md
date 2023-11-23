### ImageNet-1K / ImageNet-100
Download [ILSVRC2014](https://image-net.org/challenges/LSVRC/2014/index.php)

See [this list](./misc/imagenet100.txt) of ImageNet-100 subset.

### MSCOCO
Download `train2017` images from [MSCOCO](https://cocodataset.org/#download)

### Pascal VOC 2012
Download validation images from Pascal VOC 2012.

We organize our data as follows:
```
$HOME/VAST/data
   |------------- ILSVRC2014/
   |                |---------- Img/
   |                |            |------ train
   |                |            |------ val
   |                |
   |                |---------- Img-100/ 
   |                             |------ train
   |                             |------ val
   |
   |------------- coco/train/train2017
   |------------- voc12/val/
```
