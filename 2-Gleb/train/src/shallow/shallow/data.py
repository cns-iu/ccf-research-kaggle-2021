# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,shallow//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% tags=["active-ipynb"]
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %% [markdown]
# # Imports

# %%
from pathlib import Path
from functools import lru_cache, partial

from tqdm.auto import tqdm

import os
import cv2
import yaml
from PIL import Image
import numpy as np
import albumentations as albu
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset as ConcatDataset

from shallow import utils


# %% tags=["active-ipynb"]
# from tqdm.notebook import tqdm
# from pprint import pprint

# %% [markdown]
# # Code

# %% [markdown]
# ## Datasets

# %%
class Dataset:
    def __init__(self, root, pattern):
        self.root = Path(root)
        self.pattern = pattern
        self.files = sorted(list(self.root.glob(self.pattern)))
        self._is_empty('There is no matching files!')
        
    def apply_filter(self, filter_fn):
        self.files = filter_fn(self.files)
        self._is_empty()

    def _is_empty(self, msg='There is no item in dataset!'): assert len(self.files) > 0
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return self.process_item(self.load_item(idx))
    def load_item(self, idx): raise NotImplementedError
    def process_item(self, item): return item
#     def __add__(self, other):
#         return ConcatDataset([self, other])
    
class ImageDataset(Dataset):
    def load_item(self, idx):
        img_path = self.files[idx]
        img = Image.open(str(img_path))
        return img
    
class PairDataset:
    def __init__(self, ds1, ds2):
        self.ds1, self.ds2 = ds1, ds2
        self.check_len()
    
    def __len__(self): return len(self.ds1)
    def check_len(self): assert len(self.ds1) == len(self.ds2)
    
    def __getitem__(self, idx):
        return self.ds1.__getitem__(idx), self.ds2.__getitem__(idx) 
    
    
class TransformDataset:
    def __init__(self, dataset, transforms, is_masked=False):
        self.dataset = dataset
        self.transforms = albu.Compose([]) if transforms is None else transforms
        self.is_masked = is_masked
    
    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        if self.is_masked:
            img, mask = item
            augmented = self.transforms(image=img, mask=mask)
            return augmented["image"], augmented["mask"]
        else:
            return self.transforms(image=item[0], mask=None)['image']
    
    def __len__(self):
        return len(self.dataset)
    
class MultiplyDataset:
    def __init__(self, dataset, rate):
        _dataset = ConcatDataset([dataset])
        for i in range(rate-1):
            _dataset += ConcatDataset([dataset])
        self.dataset = _dataset
        
    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
    
    def __len__(self):
        return len(self.dataset)
    
class CachingDataset:
    def __init__(self, dataset):
        self.dataset = dataset
            
    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
    
    def __len__(self):
        return len(self.dataset)

    
class PreloadingDataset:
    def __init__(self, dataset, num_proc=False, progress=None):
        self.dataset = dataset
        self.num_proc = num_proc
        self.progress = progress
        if self.num_proc:
            self.data = utils.mp_func_gen(self.preload_data,
                                             range(len(self.dataset)),
                                             n=self.num_proc,
                                             progress=progress)
        else:
            self.data = self.preload_data(range(len(self.dataset)))
        
    def preload_data(self, args):
        idxs = args
        data = []
        if self.progress is not None and not self.num_proc: idxs = self.progress(idxs)
        for i in idxs:
            r = self.dataset.__getitem__(i)
            data.append(r)
        return data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    
class GpuPreloadingDataset:
    def __init__(self, dataset, devices):
        self.dataset = dataset
        self.devices = devices
        self.data = self.preload_data()
        
    def preload_data(self):
        data = []
        for i in range(len(self.dataset)):
            item, idx = self.dataset.__getitem__(i)
            item = item.to(self.devices[0])
            data.append((item, idx))
        return data
    
    def __getitem__(self, idx):
        return self.data[idx]
   
    def __len__(self):
        return len(self.dataset)


# %% [markdown]
# ## Dataset catalog

# %%
        


# %% [markdown]
# ## Builders

# %%
# dataset_factories = {'termit':TermitDataset}
# transform_factories = {'TRAIN':{'factory':TransformDataset_Partial_HARD, 'transform_getter':get_aug}}
# extend_factories = {'GPU_PRELOAD':GpuPreloadingDataset_Partial_GPU0}
# dataset_types = ['TRAIN', 'VALID', 'TEST']
# datasets = {'TRAIN': dataset1, 'VALID': ...}

def extend_dataset(ds, data_field, extend_factories):
    for k, factory in extend_factories.items():
        field_val = data_field.get(k, None) 
        if field_val:
            args = {}
            if isinstance(field_val, dict): args.update(field_val)
            ds = factory(ds, **args)
    return ds

def create_extensions(cfg, datasets, extend_factories):
    extended_datasets = {}
    for kind, ds in datasets.items():
        extended_datasets[kind] = extend_dataset(ds, cfg.DATA[kind], extend_factories)
    return extended_datasets

def create_datasets(cfg,
                    all_datasets,
                    dataset_types=['TRAIN', 'VALID', 'TEST']):

    converted_datasets = {}
    for dataset_type in dataset_types:
        data_field = cfg.DATA[dataset_type]
        datasets_strings = data_field.DATASETS
        
        if datasets_strings:
            datasets = [all_datasets[ds] for ds in datasets_strings]
            ds = ConcatDataset(datasets) if len(datasets)>1 else datasets[0] 
            converted_datasets[dataset_type] = ds
    return converted_datasets


def create_transforms(cfg,
                      transform_factories,
                      dataset_types=['TRAIN', 'VALID', 'TEST']):
    transformers = {}
    for dataset_type in dataset_types:
        aug_type = cfg.TRANSFORMERS[dataset_type]['AUG']
        args={
            'aug_type':aug_type,
            'transforms_cfg':cfg.TRANSFORMERS
        }
        if transform_factories[dataset_type]['factory'] is not None:
            transform_getter = transform_factories[dataset_type]['transform_getter'](**args)
            transformer = partial(transform_factories[dataset_type]['factory'], transforms=transform_getter)
        else:
            transformer = lambda x: x
        transformers[dataset_type] = transformer
    return transformers    

def apply_transforms_datasets(datasets, transforms):
    return {dataset_type:transforms[dataset_type](dataset) for dataset_type, dataset in datasets.items()}



# %%
def build_dataloaders(cfg, datasets, samplers=None, batch_sizes=None, num_workers=1, drop_last=False, pin=False):
    # TODO DDP logic
    dls = {}
    for kind, dataset in datasets.items():
        sampler = samplers[kind]    
        shuffle = kind == 'TRAIN' if sampler is None else False
        batch_size = batch_sizes[kind] if batch_sizes[kind] is not None else 1
        dls[kind] = create_dataloader(dataset, sampler, shuffle, batch_size, num_workers, drop_last, pin)
            
    return dls
    
def create_dataloader(dataset, sampler=None, shuffle=False, batch_size=1, num_workers=1, drop_last=False, pin=False):
    collate_fn=None
    assert not(sampler is not None and shuffle)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    return data_loader

# %% [markdown]
# # Tests

# %% [markdown]
# ## test datasets

# %% tags=["active-ipynb"]
# imgs_path = './test_data/'

# %% [markdown]
# ### 1

# %% tags=["active-ipynb"]
# d1 = Dataset(imgs_path, 'aimg*.png')
# assert len(d1) == 1
# try:
#     d1[0]
# except NotImplementedError:
#     pass
# except Exception as e:
#     raise e

# %% [markdown]
# ### 2

# %% tags=["active-ipynb"]
# d2 = ImageDataset(imgs_path, 'aimg_*')
# d2.process_item = lambda x: np.array(x)
# assert len(d2) == 1
# d3 = ImageDataset(imgs_path, 'mask_*')
# d3.process_item = lambda x: np.array(x)
# assert len(d3) == 1
# d2[0].shape, d3[0].shape

# %% [markdown]
# ### 3

# %% tags=["active-ipynb"]
# d4 = PairDataset(d2, d3)
# assert len(d4) == 1
# i,ii = d4[0]
# j, jj = d2[0], d3[0]
# np.allclose(i,j), np.allclose(ii,jj)

# %% [markdown]
# ### transforms dataset

# %% tags=["active-ipynb"]
# transforms = albu.Compose([albu.CenterCrop(50, 50)])
# d5 = TransformDataset(d4, transforms=transforms, is_masked=True)
# i = d5[0]
# i[0].shape, i[0].shape

# %% [markdown]
# ### multiply

# %% tags=["active-ipynb"]
# mult = 2
# d6 = MultiplyDataset(d2, mult)
# assert len(d6) // mult == len(d2)

# %% [markdown]
# ### cache

# %% tags=["active-ipynb"]
# d7 = CachingDataset(d2)

# %% tags=["active-ipynb"]
# %%timeit -r 10 -n 100
# d7[0]

# %% tags=["active-ipynb"]
# %%timeit -r 1 -n 5
# d2[1]

# %% [markdown]
# ### preloading

# %% tags=["active-ipynb"]
# _d8 = ImageDataset(imgs_path, 'aimg_9*.png')
# d8 = PreloadingDataset(_d8, num_proc=8)
# assert len(_d8) == len(d8)

# %% tags=["active-ipynb"]
# %%timeit -r 10 -n 100
# d8[18]

# %% [markdown]
# ## test catalog

# %% tags=["active-ipynb"]
# class MyDatasetCatalog(DatasetCatalog):
#     DATA_DIR = "test_data/"
#     DATA_DIR_MNT = "/mnt/input/term"
#     
#     DATASETS = {
#         "test_data": {
#                         'factory':'factory_test',
#                         'path_args':{
#                                         "root": "validation_1_1",
#                                     },
#                         'kwargs':{
#                                         "pattern": 'aimg*.png'
#                                 }
#             
#         },
#         "test_data_masks": {
#             'factory':'factory_test_masks',
#             'path_args':{
#                     "root": "validation_1_1",
#             },
#             'kwargs':{
#                     "pattern": 'aimg*.png'
#             }
#         },
#         "test_data_joined": {
#             'factory':'factory_test_joined',
#             'path_args':{
#                     "root1": "validation_1_1",
#                     "root2": "validation_1_1",
#             },
#             'kwargs':{
#                     "pattern1": 'aimg*.png',
#                     "pattern2": 'mask*.png'
#             }
#         }
#     }
#     
#     @staticmethod
#     def get(name): return super(MyDatasetCatalog, MyDatasetCatalog).get(name)
#     
#     @staticmethod
#     def create_factory_dict(data_dir, dataset_attrs):
#         factory = dataset_attrs['factory']
#         allowed_facts = [v['factory']  for v in MyDatasetCatalog.DATASETS.values()]
#         if factory not in allowed_facts: raise RuntimeError(f' Uknnown factory type: {factory}' )
#         
#         path_args = {k:os.path.join(data_dir, v) for k, v in dataset_attrs['path_args'].items()}
#         return dict(factory=factory, args={**path_args, **dataset_attrs['kwargs']})

# %% tags=["active-ipynb"]
# test_fact_args = MyDatasetCatalog.get(name='test_data_masks')
# test_fact_args

# %% tags=["active-ipynb"]
# test_fact_args = MyDatasetCatalog.get(name='test_data_joined')
# test_fact_args

# %% [markdown]
# ## builders

# %% tags=["active-ipynb"]
# from nb_configer import cfg

# %% tags=["active-ipynb"]
# yaml_str = '''
#     DATA:
#       TRAIN:
#         DATASETS: ['test_data_joined', 'test_data_joined']
#         GPU_PRELOAD: False
#         PRELOAD: True
#         CACHE: False
#       VALID:
#         DATASETS: ['test_data']
#       TEST:
#         DATASETS: ['test_data']
#
#     TRANSFORMERS:
#       TRAIN:
#         AUG: 'test'
#       VALID:
#         AUG: 'val'
#       TEST:
#         AUG: 'test'
#
#       CROP: [256, 256]
#       RESIZE: [512, 512]
#
#     TRAIN:
#       NUM_WORKERS: 0
#       BATCH_SIZE: 128
#
#     VALID:
#       NUM_WORKERS: 4
#       BATCH_SIZE: 1
#     '''
# yd = yaml.safe_load(yaml_str)
# with open('/tmp/t.yaml', 'w') as f:
#     yaml.safe_dump(yd, f)
# cfg.merge_from_file('/tmp/t.yaml')

# %% tags=["active-ipynb"]
# pprint(cfg.DATA)

# %% tags=["active-ipynb"]
# def test_trans_get(aug_type, transforms_cfg):
#     w,h = transforms_cfg['RESIZE']
#     return albu.Compose([albu.CenterCrop(h, w)])

# %% tags=["active-ipynb"]
# class ImageDatasetArray(ImageDataset):
#     def process_item(self, item): return np.array(item)
# class PairImageDataset(PairDataset):
#     def __init__(self, root1, pattern1, root2, pattern2):
#         self.ds1 = ImageDatasetArray(root1, pattern1)
#         self.ds2 = ImageDatasetArray(root2, pattern2)
#         assert len(self.ds1) == len(self.ds2)

# %% tags=["active-ipynb"]
# dataset_factories = {'factory_test': ImageDatasetArray, 'factory_test_joined': PairImageDataset}
# transform_factory = {
#     'TRAIN':{'factory':partial(TransformDataset, is_masked=True), 'transform_getter':test_trans_get},
#     'TEST':{'factory':TransformDataset, 'transform_getter':test_trans_get},
#     'VALID':{'factory':TransformDataset, 'transform_getter':test_trans_get},
# }
# extend_factories = {
#     'GPU_PRELOAD':GpuPreloadingDataset,
#     'PRELOAD':partial(PreloadingDataset, num_proc=8),
#     'CACHE':CachingDataset,
# }

# %% tags=["active-ipynb"]
# datasets = create_datasets(cfg, MyDatasetCatalog, dataset_factories)
# datasets = create_extensions(cfg, datasets, extend_factories)
#
# transforms = create_transforms(cfg, transform_factory)
# datasets = apply_transforms_datasets(datasets, transforms)

# %%

# %%

# %%
