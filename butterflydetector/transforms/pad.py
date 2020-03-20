import copy
import logging

import torchvision
import PIL

from .preprocess import Preprocess
from .scale import _scale
import math

LOG = logging.getLogger(__name__)


class CenterPad(Preprocess):
    def __init__(self, target_size):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, anns, ltrb = self.center_pad(image, anns)
        meta['offset'] -= ltrb[:2]

        LOG.debug('valid area before pad with %s: %s', ltrb, meta['valid_area'])
        meta['valid_area'][:2] += ltrb[:2]
        LOG.debug('valid area after pad: %s', meta['valid_area'])

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta

    def center_pad(self, image, anns):
        w, h = image.size

        left = int((self.target_size[0] - w) / 2.0)
        top = int((self.target_size[1] - h) / 2.0)
        if left < 0:
            left = 0
        if top < 0:
            top = 0

        right = self.target_size[0] - w - left
        bottom = self.target_size[1] - h - top
        if right < 0:
            right = 0
        if bottom < 0:
            bottom = 0
        ltrb = (left, top, right, bottom)

        # pad image
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=(124, 116, 104))

        # pad annotations
        for ann in anns:
            ann['keypoints'][:, 0] += ltrb[0]
            ann['keypoints'][:, 1] += ltrb[1]
            ann['bbox'][0] += ltrb[0]
            ann['bbox'][1] += ltrb[1]
            if 'class' in ann.keys():
                ann['class'][:, 0] += ltrb[0]
                ann['class'][:, 1] += ltrb[1]

        return image, anns, ltrb

class CenterPadTight(Preprocess):
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, anns, ltrb = self.center_pad(image, anns)
        meta['offset'] -= ltrb[:2]

        LOG.debug('valid area before pad with %s: %s', ltrb, meta['valid_area'])
        meta['valid_area'][:2] += ltrb[:2]
        LOG.debug('valid area after pad: %s', meta['valid_area'])

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta

    def center_pad(self, image, anns):
        w, h = image.size
        target_width = math.ceil((w) / self.multiple) * self.multiple #+ 1
        target_height = math.ceil((h) / self.multiple) * self.multiple #+ 1

        left = int((target_width - w) / 2.0)
        top = int((target_height - h) / 2.0)
        if left < 0:
            left = 0
        if top < 0:
            top = 0

        right = target_width - w - left
        bottom = target_height - h - top
        if right < 0:
            right = 0
        if bottom < 0:
            bottom = 0
        ltrb = (left, top, right, bottom)

        # pad image
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=(124, 116, 104))

        # pad annotations
        for ann in anns:
            ann['keypoints'][:, 0] += ltrb[0]
            ann['keypoints'][:, 1] += ltrb[1]
            ann['bbox'][0] += ltrb[0]
            ann['bbox'][1] += ltrb[1]

        return image, anns, ltrb

class CenterPadTightConditional(Preprocess):
    def __init__(self, multiple, res_threshold, resample=PIL.Image.BICUBIC):
        self.multiple = multiple
        self.res_threshold = res_threshold
        self.resample = resample

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)
        w, h = image.size
        if not(w>self.res_threshold or h>self.res_threshold):
            image, anns, ltrb = self.center_pad(image, anns)
            meta['offset'] -= ltrb[:2]

            LOG.debug('valid area before pad with %s: %s', ltrb, meta['valid_area'])
            meta['valid_area'][:2] += ltrb[:2]
            LOG.debug('valid area after pad: %s', meta['valid_area'])

            for ann in anns:
                ann['valid_area'] = meta['valid_area']
        else:
            this_long_edge = 8320
            s = this_long_edge / max(h, w)
            if h > w:
                target_w, target_h = int(w * s), this_long_edge
                target_w = -((-target_w) // self.multiple) * self.multiple
            else:
                target_w, target_h = this_long_edge, int(h * s)
                target_h = -((-target_h) // self.multiple) * self.multiple
            image, anns, meta = _scale(image, anns, meta, target_w, target_h, self.resample)
        return image, anns, meta

    def center_pad(self, image, anns):
        w, h = image.size
        target_width = math.ceil((w ) / self.multiple) * self.multiple #+ 1
        target_height = math.ceil((h ) / self.multiple) * self.multiple #+ 1

        left = int((target_width - w) / 2.0)
        top = int((target_height - h) / 2.0)
        if left < 0:
            left = 0
        if top < 0:
            top = 0

        right = target_width - w - left
        bottom = target_height - h - top
        if right < 0:
            right = 0
        if bottom < 0:
            bottom = 0
        ltrb = (left, top, right, bottom)

        # pad image
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=(124, 116, 104))

        # pad annotations
        for ann in anns:
            ann['keypoints'][:, 0] += ltrb[0]
            ann['keypoints'][:, 1] += ltrb[1]
            ann['bbox'][0] += ltrb[0]
            ann['bbox'][1] += ltrb[1]

        return image, anns, ltrb

class SquarePad(Preprocess):
    def __call__(self, image, anns, meta):
        center_pad = CenterPad(max(image.size))
        return center_pad(image, anns, meta)
