import torch
import torch.nn as nn

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



class TextFeatures():
    def __init__(self, clip, tokenizer, classname_c, classname_f):
        self.clip = clip
        self.tokenizer = tokenizer
        
        self.classname_c = classname_c
        self.classname_f = classname_f

        self.features_c = self.get_text_features(classname_c + ["Background"])
        self.features_f = [self.get_text_features([c for c in classname_f if sup_c in c]) for sup_c in classname_c]

    def get_text_features(self, vocab):
        text = ['a photo of a {}'.format(c) for c in vocab]
        text = torch.cat([self.tokenizer(c) for c in text], dim=0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip.encode_text(text.cuda()).cpu().float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features


def get_masked_pred_c(model, text_features, image, mask_c):    
    pred_c = torch.zeros_like(mask_c)
    
    for i in range(int(mask_c.max().item()) + 1):
        masked_image = image.clone()
        masked_image[(mask_c != i).repeat(3,1,1)] = 0.5
        masked_image = masked_image.unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(masked_image.cuda()).cpu().float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            t = text_features.features_c
            text_probs = (image_features @ t.T).softmax(dim=-1)
            p = text_probs.max(dim=1)[1].item()
            pred_c[mask_c == i] = p

    return pred_c


def get_masked_pred_f(model, text_features, image, mask_f, pred_c):    
    pred_f = torch.zeros_like(mask_f)
    
    for i in range(int(mask_f.max().item()) + 1):
        masked_image = image.clone()
        masked_image[(mask_f != i).repeat(3,1,1)] = 0.5
        masked_image = masked_image.unsqueeze(0).cuda()

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(masked_image.cuda()).cpu().float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            loc = (mask_f == i).nonzero()[0]
            cls = int(pred_c[tuple(loc)].item())
            if cls < len(text_features.classname_c):
                t = text_features.features_f[cls]
                text_probs = (image_features @ t.T).softmax(dim=-1)
                p = text_probs.max(dim=1)[1].item()
                p = sum([len(text_features.features_f[i]) for i in range(cls)]) + p
            else:
                p = len(text_features.classname_f)
            pred_f[mask_f == i] = p

    return pred_f


def get_masked_pred_sam_c(model, text_features, image, mask_c):    
    pred_c = -torch.ones_like(mask_c[0]).int()
    nc, nf = 11, 40  # number of object and part classes
    
    for m in mask_c:
        masked_image = image.clone()
        masked_image[~m.repeat(3,1,1)] = 0.5
        masked_image = masked_image.unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(masked_image.cuda()).cpu().float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            t = text_features.features_c
            text_probs = (image_features @ t.T).softmax(dim=-1)
            p = text_probs.max(dim=1)[1].item()
            pred_c[m.logical_and(pred_c == -1)] = p

    pred_c[pred_c == -1] = nc
    return pred_c


def get_masked_pred_sam_f(model, text_features, image, mask_f, pred_c):    
    pred_f = -torch.ones_like(mask_f[0]).int() 
    nc, nf = 11, 40  # number of object and part classes
    
    for m in mask_f:
        masked_image = image.clone()
        masked_image[~m.repeat(3,1,1)] = 0.5
        masked_image = masked_image.unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(masked_image.cuda()).cpu().float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            loc = m.nonzero()[0]
            cls = int(pred_c[tuple(loc)].item())
            if cls < nc:
                t = text_features.features_f[cls]
                text_probs = (image_features @ t.T).softmax(dim=-1)
                p = text_probs.max(dim=1)[1].item()
                p = sum([len(text_features.features_f[i]) for i in range(cls)]) + p
            else:
                p = nf
            pred_f[m] = p

    pred_f[pred_f == -1] = nf
    return pred_f


def visualize_img(image_tensor, normalize):
    mean = torch.tensor(normalize.mean).view(-1, 1, 1)
    std= torch.tensor(normalize.std).view(-1, 1, 1)
    image_tensor = image_tensor * std + mean
    image_np = image_tensor.numpy()
    image_np = (image_np * 255).astype(int)
    image_np = image_np.transpose(1, 2, 0)

    plt.figure()
    plt.axis('off')
    plt.imshow(image_np)


def visualize_seg(image_tensor, cmap):    
    image_np = image_tensor.int().numpy()[0]
    
    plt.figure()
    plt.axis('off')
    plt.imshow(image_np, cmap=cmap, vmin=0, vmax=len(cmap.colors)-1, interpolation='nearest')
    
    
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def create_colormap(names):    
    colors = plt.get_cmap("tab10").colors

    cmap_c = [(0,0,0) for _ in range(12)]
    cmap_f = [(0,0,0) for _ in range(41)]
    for i in range(11):
        if i < 4:
            offset = 0
        elif i == 8:
            offset = 3
        elif i == 10:
            offset = 4
        else:
            continue

        lens = [len(names[k]) for k in names.keys()]
        base_color = colors[i-offset]
        base_index = sum([lens[_] for _ in range(i)])

        cmap_c[i] = base_color
        for j in range(lens[i]):
            amount = np.arange(0.5, 2.1, 0.3)[j]
            cmap_f[base_index + j] = adjust_lightness(base_color, amount=amount)

    cmap_c = ListedColormap(cmap_c)
    cmap_f = ListedColormap(cmap_f)
    return cmap_c, cmap_f
