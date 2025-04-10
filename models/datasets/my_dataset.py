import logging
import os
import pathlib

import cv2
import torch
import numpy as np


class MyDataset:

    def __init__(self, root, split, transform=None, target_transform=None, is_test=False, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.split = pathlib.Path(split)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.split / "test.txt"
        else:
            image_sets_file = self.split / "train.txt"
        self.keep_difficult = keep_difficult
        
        label_file_name = self.split / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip() + ","

            # classes should be a comma separated list

            classes = class_string.split(',')[:-1]
            classes = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND', 'face')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.info = self._read_image_info(image_sets_file)

    def __getitem__(self, index):
        image_info = self.info[index].copy()
        path, boxes, labels = image_info["path"], image_info["annos"], image_info["labels"]
        image = self._read_image(path)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        if self.target_transform and (torch.any(torch.isnan(boxes)) or torch.any(torch.isinf(boxes))):
            print("nan or inf in boxes")
            if torch.any(torch.isinf(boxes)):
                inf_rows = torch.any(torch.isinf(boxes), dim=(1, 2))  
                inf_indices = torch.nonzero(inf_rows).squeeze() 
            else:
                nan_rows = torch.any(torch.isnan(boxes), dim=1)  
                nan_indices = torch.nonzero(nan_rows).squeeze()
            raise ValueError(f"nan or inf in boxes at index {index}") 
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.info[index]["path"]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        return self.info[index]["annos"]

    def __len__(self):
        return len(self.info)

    def _read_image_info(self, image_sets_file):
        info = []
        with open(image_sets_file) as f:
            for line in f:
                path, anno, label = line.split(",")
                annos_t = [list(map(int, row.strip().split())) for row in anno.split(";")]
                
                flag = True
                annos = []
                for anno in annos_t:
                    if anno[2] - anno[0] < 5 or anno[3] - anno[1] < 5:
                        continue
                    else:
                        flag = False
                        annos.append(anno)
                if flag:
                    continue
                
                label = self.class_dict[label.strip()]
                img_info = {
                    "path": path.strip(),
                    "annos": np.array(annos, dtype=np.float32),
                    "labels": np.array([label] * len(annos), dtype=np.int64),
                }
                info.append(img_info)
        return info

    def _read_image(self, path):
        image_file = self.root / path
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
