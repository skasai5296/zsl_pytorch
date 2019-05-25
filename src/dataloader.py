import sys, os
from functools import reduce

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def pil_loader(path):
    return Image.open(path).convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

class AnimalsWithAttributes(Dataset):
    """
    self.classlist : list of names for classes
    self.subclasslist : list of names for {seen, unseen} classes for mode {train|val, test}
    self.attrlist : list of names for attributes
    self.cls2attr : numpy.array of (num_classes x num_attrs), binary
    """
    def __init__(self, root_dir, mode='train', transforms=None, loader=get_default_image_loader()):
        self.mode = mode
        assert self.mode in ['train', 'val', 'test']
        self.transforms = transforms

        class_dir = os.path.join(root_dir, "classes.txt")
        with open(class_dir, "r") as f:
            classlist = f.readlines()
        self.classlist = [classname.strip().split()[1] for classname in classlist]

        if self.mode in ['train', 'val']:
            subclass_dir = os.path.join(root_dir, "trainclasses.txt")
        else:
            subclass_dir = os.path.join(root_dir, "testclasses.txt")
        with open(subclass_dir, "r") as f:
            subclasslist = f.readlines()
        self.subclasslist = [classname.strip() for classname in subclasslist]

        attr_dir = os.path.join(root_dir, "predicates.txt")
        with open(attr_dir, "r") as f:
            attrlist = f.readlines()
        self.attrlist = [attrname.strip().split()[1] for attrname in attrlist]

        matrix_dir = os.path.join(root_dir, "predicate-matrix-binary.txt")
        with open(matrix_dir, "r") as f:
            matrix = f.readlines()
        self.cls2attr = np.array([[int(i) for i in onehot.rstrip().split()] for onehot in matrix])

        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.paths = []
        self.sizes = []
        for idx, cl in enumerate(self.subclasslist):
            filelist = os.listdir(os.path.join(self.img_dir, cl))
            self.sizes.append(len(filelist))
            self.paths.extend([os.path.join(self.img_dir, cl, file) for file in filelist])
        self.len = len(self.paths)
        self.ranges = np.cumsum(self.sizes)
        print("{} dataset loaded, {} images".format(self.mode, self.len))

        self.loader = loader
        self.transforms = transforms


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.loader(self.paths[idx])
        onehot = np.zeros(len(self.classlist))
        for i, num in enumerate(self.ranges):
            if idx < num:
                cl = i
                onehot[i] = 1
                break
        att = self.cls2attr[cl]
        return {'image': img, 'attributes': att, 'class': onehot}
        pass

class SUN_Attributes(Dataset):
    '''
    SUN Attributes dataset class
    root_dir : root directory
    img_dir : directory of image files
    ann_dir : directory of annotation
    transform : how to transform images, should be from torchvision.transforms
    train : True for training
    '''
    def __init__(self, root_dir, image_dir, ann_dir, transform=None, train=True):
        img_dir = os.path.join(root_dir, image_dir)
        att_dir = os.path.join(root_dir, ann_dir, 'attributeLabels_continuous.mat')
        imgn_dir = os.path.join(root_dir, ann_dir, 'images.mat')
        attn_dir = os.path.join(root_dir, ann_dir, 'attributes.mat')
        self.attrs = loadmat(att_dir)['labels_cv']
        self.attrnames = loadmat(attn_dir)['attributes']
        self.attrnames = [i[0] for i in self.attrnames]
        self.imgs = loadmat(imgn_dir)['images']
        self.imgs = [os.path.join(img_dir, i[0]) for i in self.imgs]
        self.len = len(self.imgs)
        self.feature_size = len(self.attrnames)
        self.idx = np.arange(self.len)
        np.random.shuffle(self.idx)
        if train:
            self.dataidx = self.idx[:self.len*4//5]
        else:
            self.dataidx = self.idx[self.len*4//5:]
        self.transform = transform

        print('completed loading dataset (Training = {}), length is {}'.format(train, self.len), flush=True)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        impath = self.imgs[idx]
        img = Image.open(impath)
        img = img.convert('RGB')
        ann = self.attrs[idx]

        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'attributes': ann}

        return sample


if __name__ == '__main__':
    dset = AnimalsWithAttributes("../dataset/Animals_with_Attributes2", mode='test')
    print(dset[0])
