#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
# import lmdb
import six
import sys
from PIL import Image, ImageOps
import numpy as np


class listDataset(Dataset):
    def __init__(self, list_file=None, transform=None, target_transform=None):
        self.list_file = list_file
        with open(list_file) as fp:
            self.lines = fp.readlines()
            self.nSamples = len(self.lines)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        line_splits = self.lines[index].strip().split(' ')
        imgpath = line_splits[0]
        
        try:
            if 'train' in self.list_file:
                ##默认是需要转成L单通道
                #1D
                #img = Image.open(imgpath).convert('L')
                #2D
                img = Image.open(imgpath)
            else:
                #1D
                #img = Image.open(imgpath).convert('L')
                #2D
                img = Image.open(imgpath)
        except IOError:
            print('imgpath:',imgpath)
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        label = line_splits[1]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class paddingNormalize(object):

    def __init__(self, imgh, imgw, padding_value=0):
        self.imgh = imgh
        self.imgw = imgw
        self.padding_value = padding_value
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):

        w, h = img.size
        top = int((self.imgh - h)/2)
        bottom = self.imgh - top -h
        left = int((self.imgw -w)/2)
        right = self.imgw - left - w
        img=ImageOps.expand(img, border=(left,top,right,bottom), fill=self.padding_value)##left,top,right,bottom
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        #img = img.permute(0,2,1)
        return img

class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        ##如果keep ratio，表示一个batch中取w/h的最大值缩放整个batch中的图片
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
        
        ##1D
        # transform = resizeNormalize((imgW, imgH))
        # images = [transform(image) for image in images]
        
        ##2D
        transform = paddingNormalize(imgH, imgW, padding_value=0)
        images = [transform(image) for image in images]

        #unsqueeze(0)表示增加第一维，组成batch维度
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


