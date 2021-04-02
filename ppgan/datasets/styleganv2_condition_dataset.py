#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import random
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image

from paddle.io import Dataset
from paddle.vision.datasets import ImageFolder

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register()
class StyleGANv2ConditionDataset(BaseDataset):
    def __init__(self, target_path, condition_paths):
        super().__init__()
        
        self.target_dataset = ImageFolder(target_path)
        self.condition_datasets = []
        if not isinstance(condition_paths, (list, tuple)):
            condition_paths = [condition_paths]
        for path in condition_paths:
            self.condition_datasets.append(ImageFolder(path, extensions='.npy'))
        
    def __len__(self):
        return len(self.target_dataset)
        
    def __getitem__(self, index):
        target = np.asarray(Image.open(self.target_dataset.samples[index])).transpose([2,0,1]).astype('float32') / 255 *  2 - 1
        condition = np.load(random.choice(self.condition_datasets).samples[index]).astype('float32') / 255 *  2 - 1
        return {
            'A': target,
            'condition': condition
        }

    def prepare_data_infos(self, dataroot):
        pass
