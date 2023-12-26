"""Script for training softmax classifier only.
"""

from __future__ import print_function, division
import os

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.parallel.scatter_gather as scatter_gather
import tensorboardX
from tqdm import tqdm

from lib.nn.parallel.data_parallel import DataParallel
from lib.nn.optimizer import SGD
from lib.nn.sync_batchnorm.batchnorm import convert_model
from lib.nn.sync_batchnorm.replicate import patch_replication_callback
from spml.config.default import config
from spml.config.parse_args import parse_args
import spml.utils.general.train as train_utils
import spml.utils.general.vis as vis_utils
import spml.utils.general.others as other_utils
import spml.models.utils as model_utils
from spml.data.datasets.spix_seeds_base_dataset import SPixListDataset
from spml.models.embeddings.resnet_pspnet import resnet_50_pspnet, resnet_101_pspnet
from spml.models.embeddings.resnet_deeplab import resnet_50_deeplab, resnet_101_deeplab
import spml.models.embeddings.cast as cast
from spml.models.predictions.softmax_classifier import softmax_classifier, linear_classifier

torch.cuda.manual_seed_all(235)
torch.manual_seed(235)

cudnn.enabled = True
cudnn.benchmark = True


def main():
  """Training for softmax classifier.
  """
  # Retreve experiment configurations.
  args = parse_args('Training for softmax classifier.')

  # Retrieve GPU informations.
  device_ids = [int(i) for i in config.gpus.split(',')]
  gpu_ids = [torch.device('cuda', i) for i in device_ids]
  num_gpus = len(gpu_ids)

  # Create logger and tensorboard writer.
  summary_writer = tensorboardX.SummaryWriter(logdir=args.snapshot_dir)
  color_map = vis_utils.load_color_map(config.dataset.color_map_path)

  model_path_template = os.path.join(args.snapshot_dir,
                                     'model-{:d}.pth')
  optimizer_path_template = os.path.join(args.snapshot_dir,
                                         'model-{:d}.state.pth')

  # Create data loaders.
  train_dataset = SPixListDataset(
      data_dir=args.data_dir,
      data_list=args.data_list,
      img_mean=config.network.pixel_means,
      img_std=config.network.pixel_stds,
      size=config.train.crop_size,
      random_crop=config.train.random_crop,
      random_scale=config.train.random_scale,
      random_mirror=config.train.random_mirror,
      training=True,
      n_segments=config.network.num_superpixels,
      slic_compactness=10.0,
      slic_scale_factor=0.5)

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=config.train.batch_size,
      shuffle=config.train.shuffle,
      num_workers=num_gpus * config.num_threads,
      drop_last=False,
      collate_fn=train_dataset.collate_fn)

  # Create models.
  if config.network.backbone_types == 'panoptic_pspnet_101':
    embedding_model = resnet_101_pspnet(config).cuda()
  elif config.network.backbone_types == 'panoptic_deeplab_101':
    embedding_model = resnet_101_deeplab(config).cuda()
  elif config.network.backbone_types == 'cast_small_p512_finetune':
    embedding_model = cast.cast_small_p512_finetune(config).cuda()
  else:
    raise ValueError('Not support ' + config.network.backbone_types)

  if config.network.prediction_types == 'softmax_classifier':
    prediction_model = softmax_classifier(config).cuda()
  elif config.network.prediction_types == 'linear_classifier':
    prediction_model = linear_classifier(config).cuda()
  else:
    raise ValueError('Not support ' + config.network.prediction_types)

  # Use synchronize batchnorm.
  if config.network.use_syncbn:
    embedding_model = convert_model(embedding_model).cuda()
    prediction_model = convert_model(prediction_model).cuda()
 
  # Use customized optimizer and pass lr=1 to support different lr for
  # different weights.
  optimizer = SGD(
      embedding_model.get_params_lr() + prediction_model.get_params_lr(),
      lr=1,
      momentum=config.train.momentum,
      weight_decay=config.train.weight_decay)
  optimizer.zero_grad()

  # Load pre-trained weights.
  curr_iter = config.train.begin_iteration
  if config.train.resume:
    model_path = model_path_template.fromat(curr_iter)
    print('Resume training from {:s}'.format(model_path))
    embedding_model.load_state_dict(
        torch.load(model_path)['embedding_model'],
        resume=True)
    prediction_model.load_state_dict(
        torch.load(model_path)['prediction_model'],
        resume=True)
    optimizer.load_state_dict(torch.load(
        optimizer_path_template.format(curr_iter)))
  elif config.network.pretrained:
    print('Loading pre-trained model: {:s}'.format(config.network.pretrained))
    if 'cast' in config.network.backbone_types:
      checkpoint = torch.load(config.network.pretrained)['state_dict']
      cast.interpolate_pos_embed(embedding_model.vit_backbone, checkpoint)
      embedding_model.load_state_dict(checkpoint, resume=False)
    else:
      embedding_model.load_state_dict(torch.load(config.network.pretrained))
  else:
    print('Training from scratch')

  # Distribute model weights to multi-gpus.
  embedding_model = DataParallel(embedding_model,
                                 device_ids=device_ids,
                                 gather_output=False)
  prediction_model = DataParallel(prediction_model,
                                  device_ids=device_ids,
                                  gather_output=False)
  if config.network.use_syncbn:
    patch_replication_callback(embedding_model)
    patch_replication_callback(prediction_model)

  for module in embedding_model.modules():
    if isinstance(module, _BatchNorm) or isinstance(module, _ConvNd):
      print(module.training, module)
  print(embedding_model)
  print(prediction_model)

  # Create memory bank.
  memory_banks = {}

  # start training
  train_iterator = train_loader.__iter__()
  iterator_index = 0
  pbar = tqdm(range(curr_iter, config.train.max_iteration))
  for curr_iter in pbar:
    # Check if the rest of datas is enough to iterate through;
    # otherwise, re-initiate the data iterator.
    if iterator_index + num_gpus >= len(train_loader):
        train_iterator = train_loader.__iter__()
        iterator_index = 0

    # Feed-forward.
    image_batch, label_batch = other_utils.prepare_datas_and_labels_mgpu(
        train_iterator, gpu_ids)
    iterator_index += num_gpus

    # Generate embeddings, clustering and prototypes.
    if config.network.prediction_types == 'linear_classifier':
      with torch.no_grad():
        embeddings = embedding_model(*zip(image_batch, label_batch))
    else:
      embeddings = embedding_model(*zip(image_batch, label_batch))

    # Compute loss.
    outputs = prediction_model(*zip(embeddings, label_batch))
    outputs = scatter_gather.gather(outputs, gpu_ids[0])
    losses = []
    for k in ['sem_ann_loss', 'clustering_loss']:
      loss = outputs.get(k, None)
      if loss is not None:
        outputs[k] = loss.mean()
        losses.append(outputs[k])
    loss = sum(losses)
    acc = outputs['accuracy'].mean()

    # Write to tensorboard summary.
    writer = (summary_writer if curr_iter % config.train.tensorboard_step == 0
               else None)
    if writer is not None:
      summary_vis = []
      summary_val = {}
      # Gather labels to cpu.
      cpu_label_batch = scatter_gather.gather(label_batch, -1)
      summary_vis.append(vis_utils.convert_label_to_color(
          cpu_label_batch['semantic_label'], color_map))
      summary_vis.append(vis_utils.convert_label_to_color(
          cpu_label_batch['instance_label'], color_map))

      # Gather outputs to cpu.
      vis_names = ['embedding']
      cpu_embeddings = scatter_gather.gather(
          [{k: emb.get(k, None) for k in vis_names} for emb in embeddings],
          -1)
      for vis_name in vis_names:
        if cpu_embeddings.get(vis_name, None) is not None:
          summary_vis.append(vis_utils.embedding_to_rgb(
              cpu_embeddings[vis_name], 'pca'))

      val_names = ['sem_ann_loss', 'accuracy', 'clustering_loss']
      for val_name in val_names:
        if outputs.get(val_name, None) is not None:
          summary_val[val_name] = outputs[val_name].mean().to('cpu')

      vis_utils.write_image_to_tensorboard(summary_writer,
                                           summary_vis,
                                           summary_vis[-1].shape[-2:],
                                           curr_iter)
      vis_utils.write_scalars_to_tensorboard(summary_writer,
                                             summary_val,
                                             curr_iter)

    # Backward propogation.
    if config.train.lr_policy == 'step':
      lr = train_utils.lr_step(config.train.base_lr,
                               curr_iter,
                               config.train.decay_iterations,
                               config.train.warmup_iteration)
    else:
      lr = train_utils.lr_poly(config.train.base_lr,
                               curr_iter,
                               config.train.max_iteration,
                               config.train.warmup_iteration)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step(lr)

    # Snapshot the trained model.
    if ((curr_iter+1) % config.train.snapshot_step == 0
         or curr_iter == config.train.max_iteration - 1):
      model_state_dict = {
        'embedding_model': embedding_model.module.state_dict(),
        'prediction_model': prediction_model.module.state_dict()}
      torch.save(model_state_dict,
                 model_path_template.format(curr_iter))
      torch.save(optimizer.state_dict(),
                 optimizer_path_template.format(curr_iter))

    # Print loss in the progress bar.
    line = 'loss = {:.3f}, acc = {:.3f}, lr = {:.6f}'.format(
        loss.item(), acc.item(), lr)
    pbar.set_description(line)


if __name__ == '__main__':
  main()
