from collections import defaultdict
import copy
import logging
import os

import numpy as np
import torch.utils.data
from PIL import Image

from .. import transforms, utils


LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class Coco(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        image_dir (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
    """
    train_image_dir = '../datasets/coolingTowersV3/images/train/'
    val_image_dir = '../datasets/coolingTowersV3/images/val/'
    train_annotations = '../datasets/coolingTowersV3/images/annotations/instances_train.json'
    val_annotations = '../datasets/coolingTowersV3/images/annotations/instances_val.json'
    test_path = {'val': ['../datasets/coolingTowersV3/images/val/',
                        '../datasets/coolingTowersV3/images/annotations/instances_val.json']}



    def __init__(self, root, annFile, *, target_transforms=None,
                 n_images=None, preprocess=None,
                 category_ids=None,
                 image_filter='keypoint-annotations'):
        if category_ids is None:
            category_ids = [1]

        from pycocotools.coco import COCO  # pylint: disable=import-outside-toplevel
        self.image_dir = root
        self.coco = COCO(annFile)

        self.category_ids = category_ids

        if image_filter == 'all':
            self.ids = self.coco.getImgIds()
        elif image_filter == 'annotated':
            self.ids = self.coco.getImgIds(catIds=self.category_ids)
            self.filter_for_annotations()
        elif image_filter == 'keypoint-annotations':
            self.ids = self.coco.getImgIds(catIds=self.category_ids)
            self.filter_for_keypoint_annotations()
        else:
            raise Exception('unknown value for image_filter: {}'.format(image_filter))

        if n_images:
            self.ids = self.ids[:n_images]
        LOG.info('Images: %d', len(self.ids))

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

    def filter_for_keypoint_annotations(self):
        LOG.info('filter for keypoint annotations ...')
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        LOG.info('... done.')

    def filter_for_annotations(self):
        """removes images that only contain crowd annotations"""
        LOG.info('filter for annotations ...')
        def has_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if ann.get('iscrowd'):
                    continue
                return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_annotation(image_id)]
        LOG.info('... done.')

    def class_aware_sample_weights(self, max_multiple=10.0):
        """Class aware sampling.

        To be used with PyTorch's WeightedRandomSampler.

        Reference: Solution for Large-Scale Hierarchical Object Detection
        Datasets with Incomplete Annotation and Data Imbalance
        Yuan Gao, Xingyuan Bu, Yang Hu, Hui Shen, Ti Bai, Xubin Li and Shilei Wen
        """
        ann_ids = self.coco.getAnnIds(imgIds=self.ids, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)

        category_image_counts = defaultdict(int)
        image_categories = defaultdict(set)
        for ann in anns:
            if ann['iscrowd']:
                continue
            image = ann['image_id']
            category = ann['category_id']
            if category in image_categories[image]:
                continue
            image_categories[image].add(category)
            category_image_counts[category] += 1

        weights = [
            sum(
                1.0 / category_image_counts[category_id]
                for category_id in image_categories[image_id]
            )
            for image_id in self.ids
        ]
        min_w = min(weights)
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))
        max_w = min_w * max_multiple
        weights = [min(w, max_w) for w in weights]
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))

        return weights

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]
        LOG.debug(image_info)
        with open(os.path.join(self.image_dir, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)
        # mask valid TODO still necessary?
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        LOG.debug(meta)

        # log stats
        for ann in anns:
            if getattr(ann, 'iscrowd', False):
                continue
            if not np.any(ann['keypoints'][:, 2] > 0.0):
                continue
            STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})

        # transform targets
        if self.target_transforms is not None:
            width_height = image.shape[2:0:-1]
            anns = [t(anns, width_height) for t in self.target_transforms]

        return image, anns, meta

    def __len__(self):
        return len(self.ids)

    def write_evaluations(self, eval_class, path, total_time):
        for folder in eval_class.dict_folder.keys():
            utils.mkdir_if_missing(path)
            with open(os.path.join(path,folder+".txt"), "w") as file:
                file.write("\n".join(eval_class.dict_folder[folder]))
        n_images = len(eval_class.image_ids)

        print('n images = {}'.format(n_images))
        print('decoder time = {:.1f}s ({:.0f}ms / image)'
              ''.format(eval_class.decoder_time, 1000 * eval_class.decoder_time / n_images))
        print('total time = {:.1f}s ({:.0f}ms / image)'
              ''.format(total_time, 1000 * total_time / n_images))
