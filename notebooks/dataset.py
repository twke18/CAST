import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO


class PartImageNetWithMask(Dataset):
    def __init__(self, img_root, ano_root, transform, seg_transform):
        self.img_root = img_root
        self.ano_root = ano_root

        self.coco = COCO(self.ano_root)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.classname_c = []
        self.classname_f = []
        for _, cat in self.coco.cats.items():
            if cat['supercategory'] not in self.classname_c:
                self.classname_c.append(cat['supercategory'])
            self.classname_f.append(cat['name'])
        
        self.transform = transform
        self.seg_transform = seg_transform

    def __getitem__(self, index):
        img_id = self.ids[index]

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')
        H, W = self.coco.imgs[img_id]['height'], self.coco.imgs[img_id]['width']
    
        seg_c = np.ones([H, W]) * len(self.classname_c)  # coarse seg
        seg_f = np.ones([H, W]) * len(self.classname_f)  # fine seg

        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            try:
                cat_id = ann['category_id']
                cat_c = self.coco.cats[cat_id]['supercategory']
                cat_f = self.coco.cats[cat_id]['name']

                m = self.coco.annToMask(ann)
                seg_c[m > 0] = self.classname_c.index(cat_c)
                seg_f[m > 0] = self.classname_f.index(cat_f)
            except:
                pass

        seg_c = torch.from_numpy(seg_c).float().unsqueeze(0)
        seg_f = torch.from_numpy(seg_f).float().unsqueeze(0)

        img = self.transform(img)
        seg_c = self.seg_transform(seg_c)
        seg_f = self.seg_transform(seg_f)
        
        return img, seg_c, seg_f
        
    def __len__(self):
        return len(self.ids)


class PredictedMask(Dataset):
    def __init__(self, img_root, ano_root):
        self.img_root = img_root
        self.ano_root = ano_root

        self.coco = COCO(self.ano_root)
        self._len = len(os.listdir(self.img_root))

    def __getitem__(self, index):
        file_name = self.coco.imgs[index]['file_name'].split('.')[0]
        path = os.path.join(self.img_root, "{}.npy".format(file_name))
        with open(path, 'rb') as f:
            data = np.load(f)
        data = torch.from_numpy(data).float().unsqueeze(0)
        return data

    def __len__(self):
        return self._len

