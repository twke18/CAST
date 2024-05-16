import os
from subprocess import call

with open('./misc/imagenet100.txt', 'r') as f:
  file_list = f.read().split('\n')[:-1]
  print(len(file_list))

old_dir = './data/ILSVRC2014/Img'
old_dir = './data/ILSVRC2014/Img-100'
for split in ['train', 'val']:
  for dir_name in file_list:
    os.makedirs(os.path.join(new_dir, split), exist_ok=True)
    call(['cp', '-rv',
          os.path.join(old_dir, split, dir_name),
          os.path.join(new_dir, split, dir_name)])