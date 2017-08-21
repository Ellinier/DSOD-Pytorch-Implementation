'''
Load image/class/box from a annotation file.

The annotation file is organized as:
    image_name #obj xmin ymin xmax ymax class_index ..
'''
from __future__ import print_function

import os

import numpy as np
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from encoder import DataEncoder

from PIL import Image, ImageOps
import cv2

class ListDataset(data.Dataset):

    def __init__(self, root, list_file, train, transform):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
        '''
        self.root = root
        self.train = train

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.data_encoder = DataEncoder()

        self.transform = transform

        # normMean = [0.485, 0.456, 0.406]
        # normStd = [0.229, 0.224, 0.225]
        # normTransform = transforms.Normalize(normMean, normStd)

        # self.transform = transforms.Compose([
        # 	transforms.Scale((300, 300)),
        # 	transforms.ToTensor(),
        # 	normTransform
        # 	])

        with open(list_file) as f:
        	lines = f.readlines()
        	self.num_samples = len(lines)

        for line in lines:
        	splited = line.strip().split()
        	self.fnames.append(splited[0])

        	num_objs = int(splited[1])
        	box = []
        	label = []
        	for i in range(num_objs):
        		xmin = splited[2+5*i]
        		ymin = splited[3+5*i]
        		xmax = splited[4+5*i]
        		ymax = splited[5+5*i]
        		c = splited[6+5*i]
        		box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
        		label.append(int(c))
        	self.boxes.append(torch.Tensor(box))
        	self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''
        Load a image, and encode its bbox locations and class labels.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_target: (tensor) location targets, sized [8732,4].
          conf_target: (tensor) label targets, sized [8732,].
        '''
        # Load image and bbox locations.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]

        # Data augmentation while training.
        if self.train:
        	img, boxes, labels = self.data_augmentation(img, boxes, labels)

        # Scale bbox locations to [0, 1].
        w, h = img.size
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)

        img = self.transform(img)

        # Encode loc & conf targets.
        loc_target, conf_target = self.data_encoder.encode(boxes, labels)
        return img, loc_target, conf_target

    def __len__(self):
        return self.num_samples

    def data_augmentation(self, img, boxes, labels):
    	img, boxes = self.random_flip(img, boxes)
    	img, boxes, labels = self.random_zoom(img, boxes, labels)
    	img = self.pil_to_cv(img)
    	img = self.random_contrast(img)
    	# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    	# img = self.random_hue(img)
    	# img = self.random_saturation(img)
    	# img = self.random_brightness(img)
    	# img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    	img = self.random_color_channels(img)
    	img = self.cv_to_pil(img)
    	return img, boxes, labels

    def random_flip(self, img, boxes):
        '''
        Randomly flip the image and adjust the bbox locations.

        For bbox (xmin, ymin, xmax, ymax), the flipped bbox is:
        (w-xmax, ymin, w-xmin, ymax) or (xmin, h-ymax, xmax, h-ymin).

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].

        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped bbox locations, sized [#obj, 4].
        '''
        r = random.random
        if r < 0.33:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
        elif r < 0.66:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            h = img.height
            ymin = w - boxes[:,3]
            ymax = w - boxes[:,1]
            boxes[:,1] = ymin
            boxes[:,3] = ymax

        return img, boxes

    def random_crop(self, img, boxes, labels):
        '''
        Randomly crop the image and adjust the bbox locations.

        For more details, see 'Chapter2.2: Data augmentation' of the paper.

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) bbox labels, sized [#obj,].

        Returns:
          img: (PIL.Image) cropped image.
          selected_boxes: (tensor) selected bbox locations.
          labels: (tensor) selected bbox labels.
        '''
        imw, imh = img.size
        while True:
            min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            if min_iou is None:
                return img, boxes, labels

            for _ in range(100):
                w = random.randrange(int(0.1*imw), imw)
                h = random.randrange(int(0.1*imh), imh)

                if h > 2*w or w > 2*h:
                    continue

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                roi = torch.Tensor([[x, y, x+w, y+h]])

                center = (boxes[:,:2] + boxes[:,2:]) / 2  # [N,2]
                roi2 = roi.expand(len(center), 4)  # [N,4]
                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])  # [N,2]
                mask = mask[:,0] & mask[:,1]  #[N,]
                if not mask.any():
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))

                iou = self.data_encoder.iou(selected_boxes, roi)
                if iou.min() < min_iou:
                    continue

                img = img.crop((x, y, x+w, y+h))
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)
                return img, selected_boxes, labels[mask]

    def pil_to_cv(self, pil_image):
    	cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    	return cv_image.astype(np.float32)

    def cv_to_pil(self, cv_image):
    	pil_image = Image.fromarray(cv2.cvtColor(cv_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    	return pil_image

    def random_hue(self, hsv_image, delta=18.0):
    	if random.random() < 0.5:
    		hsv_image[:,:,0] += random.uniform(-delta, delta)
    		hsv_image[:,:,0][hsv_image[:,:,0]>360.0] -= 360.0
    		hsv_image[:,:,0][hsv_image[:,:,0]<0.0] += 360.0
    	return hsv_image

    def random_saturation(self, hsv_image, low=0.5, high=1.5):
    	if random.random() < 0.5:
    		hsv_image[:,:,1] *= random.uniform(low, high)
    		hsv_image[:,:,1][hsv_image[:,:,1]>1] = 1
    		return hsv_image

    def random_brightness(self, hsv_image, delta=32.0):
    	if random.random() < 0.5:
    		hsv_image[:,:,2] += random.uniform(-delta, delta)
    	return hsv_image

    def random_contrast(self, image, low=0.5, high=1.5):
    	if random.random() < 0.5:
    		image *= random.uniform(low, high)
    	return image

    def random_color_channels(self, image):
    	perms = ((0, 1, 2), (0, 2, 1),
    	         (1, 0, 2), (1, 2, 0),
    	         (2, 0, 1), (2, 1, 0))
    	if random.random() < 0.2:
    		swap = perms[random.randint(0, len(perms)-1)]
    		image = image[:,:,swap]
    	return image

    def random_zoom_out(self, img, boxes):
        '''
        Randomly zoom out the image and adjust the bbox locations.

        For bbox (xmin, ymin, xmax, ymax), the zoomed out bbox is:
        coef -- zoom out coefficient
        ((1-coef)*w/2 + coef*xmin, (1-coef)*h/2 + coef*ymin,
         (1-coed)*w/2 + coef*xmax, (1-coef)*h/2 + coef*ymax)

        Args:
           img: (PIL.Image) image.
           boxes: (tensor) bbox locations, sized [#obj, 4].

        Return:
          img: (PIL.Image) randomly zoomed out image.
          boxes: (tensor) randomly zoomed out bbox locations, sized [#obj, 4].
        '''
        coef = random.uniform(0.5, 1)
        w = img.width
        h = img.height

        xmin = (1-coef)*w/2 + coef*boxes[:,0]
        xmax = (1-coef)*w/2 + coef*boxes[:,2]
        ymin = (1-coef)*h/2 + coef*boxes[:,1]
        ymax = (1-coef)*h/2 + coef*boxes[:,3]
        boxes[:,0] = xmin
        boxes[:,1] = ymin
        boxes[:,2] = xmax
        boxes[:,3] = ymax

        top = int(h/2*(1-coef)/coef)
        bottom = int(h/2*(1-coef)/coef)
        left = int(w/2*(1-coef)/coef)
        right = int(w/2*(1-coef)/coef)

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
        img = cv2.resize(img, (w, h))
        return img, boxes

    def random_zoom(self, img, boxes, labels):
        r = random.random
        if r < 0.33:
            img, boxes, labels = self.random_crop(img, boxes, labels)
        elif r < 0.66:
            img, boxes = self.random_zoom_out(img, boxes)
        return img, boxes, labels
