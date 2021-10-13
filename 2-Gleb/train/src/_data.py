import random
from PIL import Image
from pathlib import Path
from functools import partial, reduce, lru_cache
from collections import defaultdict

import numpy as np
from torch.utils.data.dataset import ConcatDataset as TorchConcatDataset

import utils


"""
Templates for data.py that should not be changed

"""


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
    

class ImageDataset(Dataset):
    def load_item(self, idx):
        img_path = self.files[idx]
        img = Image.open(str(img_path))
        return img
    
class TagImageDataset(Dataset):
    """
        Returns tuple (mask, class_idx) 
    """
    def __init__(self, root, pattern, tags):
        super(TagImageDataset, self).__init__(root, pattern)
        self.tags= tags

    def load_item(self, idx):
        img_path = self.files[idx]
        img = Image.open(str(img_path))
        return img, self.tags[img_path.parent.name]
    
class PairDataset:
    def __init__(self, ds1, ds2):
        self.ds1, self.ds2 = ds1, ds2
        self.check_len()
    
    def __len__(self): return len(self.ds1)
    def check_len(self): assert len(self.ds1) == len(self.ds2)
    
    def __getitem__(self, idx):
        return self.ds1.__getitem__(idx), self.ds2.__getitem__(idx) 

class TripleDataset:
    def __init__(self, ds1, ds2, ds3):
        self.ds1, self.ds2, self.ds3 = ds1, ds2, ds3
        self.check_len()
    
    def __len__(self): return len(self.ds1)
    def check_len(self): 
        assert len(self.ds1) == len(self.ds2)
        assert len(self.ds1) == len(self.ds3)
    
    def __getitem__(self, idx):
        return self.ds1.__getitem__(idx), self.ds2.__getitem__(idx), self.ds3.__getitem__(idx)


class MultiplyDataset:
    def __init__(self, dataset, rate):
        _dataset = TorchConcatDataset([dataset])
        for i in range(rate-1):
            _dataset += TorchConcatDataset([dataset])
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

class ConcatDataset:
    """
    To avoid recursive calls (like in torchvision variant)
    """
    def __init__(self, dss):
        self.length = 0
        self.ds_map = {}
        for i, ds in enumerate(dss):
            for j in range(len(ds)):
                self.ds_map[j+self.length] = i, self.length
            self.length += len(ds)
        self.dss = dss
    
    def load_item(self, idx):
        if idx >= self.__len__(): raise StopIteration
        ds_idx, local_idx = self.ds_map[idx]
        return self.dss[ds_idx].__getitem__(idx - local_idx)
    
    def _is_empty(self, msg='There is no item in dataset!'): assert len(self.files) > 0
    def __len__(self): return self.length
    def __getitem__(self, idx): return self.load_item(idx)


class TransformDataset:
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = albu.Compose([]) if transforms is None else transforms
    
    def __getitem__(self, idx):
        img, mask = self.dataset.__getitem__(idx)
        augmented = self.transforms(image=img, mask=mask)
        amask = augmented["mask"][0]# as binary
        amask = amask.view(1,*amask.shape)

        return augmented["image"], amask
    
    def __len__(self): return len(self.dataset)


class TagTransformDataset:
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = albu.Compose([]) if transforms is None else transforms
    
    def __getitem__(self, idx):
        img, mask, tag = self.dataset.__getitem__(idx)
        augmented = self.transforms(image=img, mask=mask)
        amask = augmented["mask"][0]# as binary
        amask = amask.view(1,*amask.shape)
        return augmented["image"], amask, tag
    
    def __len__(self): return len(self.dataset)

class BorderTransformDataset:
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = albu.Compose([]) if transforms is None else transforms
    
    def __getitem__(self, idx):
        img, mask, border = self.dataset.__getitem__(idx)
        augmented = self.transforms(image=img, mask1=mask, mask2=border)
        amask = augmented["mask1"][0]# as binary
        amask = amask.view(1,*amask.shape)
        aborder = augmented["mask2"][0]# as binary
        aborder = aborder.view(1,*aborder.shape)

        return augmented["image"], amask, aborder
    
    def __len__(self): return len(self.dataset)


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


class FoldDataset:
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, idx): return self.dataset[self.idxs[idx]]


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


def expander(x):
    x = np.array(x)
    return x if len(x.shape) == 3 else np.repeat(np.expand_dims(x, axis=-1), 3, -1)

def expander_float(x):
    x = np.array(x).astype(np.float32) / 255.
    return x if len(x.shape) == 3 else np.repeat(np.expand_dims(x, axis=-1), 3, -1)

def expander_float_item(item):
    x, cl_id = item
    x = np.array(x).astype(np.float32) / 255.
    return (x, cl_id) if len(x.shape) == 3 else (np.repeat(np.expand_dims(x, axis=-1), 3, -1), cl_id)

def update_mean_std(cfg, mean, std):
    was_frozen: False
    if cfg.is_frozen():
        cfg.defrost()
        was_frozen = True
        
    cfg.TRANSFORMERS.MEAN = tuple(mean.tolist())
    cfg.TRANSFORMERS.STD = tuple(std.tolist())
    if was_frozen: cfg.freeze()

def mean_std_dataset(dataset, parts=200):
    """
    taking every step-th image of dataset, averaging mean and std on them
    Image format is uint8, 0-255
    """
    mm, ss = [],[]
    step = max(1,len(dataset) // parts)
    for j in range(len(dataset)):
        if j % step == 0:
            i = dataset[j]
            d = i[0]/255.
            mm.append(d.mean(axis=(0,1)))
            ss.append(d.std(axis=(0,1)))
            #break
    return np.array(mm).mean(0), np.array(ss).mean(0)



def make_datasets_folds(cfg, datasets, n):
    # datasets: {'TRAIN':[t1,t2,t3,t4], 'VALID':[v1,v2,v3,v4]}
    folded_datasets = [{} for _ in range(n)]
    assert len(datasets['TRAIN']) == n
    assert len(datasets['VALID']) == n
    for i, (tds, vds ) in enumerate(zip(datasets['TRAIN'], datasets['VALID'])):
        folded_datasets[i]['VALID'] = vds
        folded_datasets[i]['TRAIN'] = tds
        if 'SSL' in datasets:
            folded_datasets[i]['SSL'] = datasets['SSL']

    return folded_datasets

def make_datasets_folds_by_idx(cfg, datasets, n, shuffle=False):
    '''
        Returns [{'TRAIN':ds1, 'VALID':ds2}, {}, ...]
    '''
    folded_datasets = [{} for _ in range(n)]
    dataset = datasets['TRAIN']
    rate = cfg.DATA.TRAIN.MULTIPLY.rate
    
    fold_idxs = generate_folds_idxs(rate, dataset, n, shuffle)
    folds = [FoldDataset(dataset, idxs) for idxs in fold_idxs]

    for i in range(n):
        train_folds = [folds[j] for j in range(n) if  j!=i]
        test_fold = folds[i]
        folded_datasets[i]['VALID'] = test_fold
        folded_datasets[i]['TRAIN'] = ConcatDataset(train_folds)
        if 'SSL' in datasets:
            folded_datasets[i]['SSL'] = datasets['SSL']

    
    return folded_datasets

def generate_folds_idxs(rate, dataset, n_folds, shuffle=False, seed=42):
    act_len = len(dataset) // rate
    l = list(range(act_len))
    if shuffle: 
        random.seed(seed)
        random.shuffle(l)
    
    split_idxs = np.array_split(l, n_folds)
    splits = []
    for split_idx in split_idxs:
        split_with_dups = [split_idx + i*act_len for i in range(rate)]
        splits.append(np.concatenate(split_with_dups))
    return splits


