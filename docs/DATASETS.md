# Prepare Datasets for CAST
In general, our code base expects the data struction as follows:
```
./data
   |------------- ILSVRC2014/
   |                |---------- Img/
   |                |            |------ train/
   |                |            |------ val/
   |                |
   |                |---------- Img-100/ 
   |                             |------ train/
   |                             |------ val/
   |
   |------------- coco/
   |                |---------- train/
   |                             |------ train2017/
   |
   |------------- VOCdevkit/
   |                |---------- VOC2012/
   |                |            |------ JPEGImages/
   |                |            |------ segcls/
   |                |
   |                |---------- sbd/
   |                             |------ dataset/
   |                                      |------ clsimg/
   |
   |------------- pcontext/
   |                |---------- VOCdevkit/
   |                             |------ VOC2010/
   |
   |------------- ade20k/
   |                |---------- ADEChallengeData2016/
   |
   |------------- PartImageNet/
                    |------ annotations/
                    |            |------ val.json
                    |            |------ train.json
                    |            |------ test.json
                    |
                    |------ images/
                                 |------ val/
                                 |------ train/
                                 |------ test/
```

### ImageNet-1K / ImageNet-100
Download [ILSVRC2014](https://image-net.org/challenges/LSVRC/2014/index.php).

See [this list](./misc/imagenet100.txt) of ImageNet-100 subset.  After downloading ILSVRC2014, run the following python script for filtering out the ImageNet-100 subset:

```
> python scripts/data/filter_imagenet100.py
```

### MSCOCO
Download `train2017` images from [MSCOCO](https://cocodataset.org/#download).

### Pascal VOC 2012
Download validation images from Pascal VOC 2012.  Please follow the instruction of [SPML](https://github.com/twke18/SPML#pascal-voc-2012).

### Pascal Context
Follow the command of [Segmenter](https://github.com/rstrudel/segmenter/blob/master/segm/scripts/prepare_pcontext.py).  Run Segmenter's python script:

```
> export PYTHONPATH=$(pwd)/segmenter/
> python segmenter/segm/scripts/prepare_pcontext.py ./data
```

### ADE20K
Follow the command of [Segmenter](https://github.com/rstrudel/segmenter/blob/master/segm/scripts/prepare_ade20k.py).  Run Segmenter's python script:

```
> export PYTHONPATH=$(pwd)/segmenter/
> python segmenter/segm/scripts/prepare_ade20k.py ./data
```

### PartImageNet
Download the PartImageNet_OOD dataset from the [github](https://github.com/TACJu/PartImageNet).  Decompress the zip file and put them under `./data`