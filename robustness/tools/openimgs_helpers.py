import os
import csv
import numpy as np
import torch as ch
import torch.utils.data as data
import robustness.data_augmentation as da
from robustness import imagenet_models
from .folder import default_loader, IMG_EXTENSIONS

target_transform_oi = ch.Tensor

def load_class_desc(data_dir):
    """Returns map from cid to class name."""
    
    class_names = {}
    
    with open(os.path.join(data_dir, "metadata", "class-descriptions-boxable.csv"), newline="") as csvfile:
        for ri, row in enumerate(csv.reader(csvfile, delimiter=' ', quotechar='|')):
            cid = row[0].split(',')[0]
            cname = ' '.join([row[0].split(',')[1]] + row[1:])
            assert cid not in class_names
            class_names[cid] = cname
            
    return class_names

def get_image_annotations_mode(class_names, data_dir, mode="train"):
    """Returns map from img number to label (along with verification
       source and confidence)"""
    
    assert mode in set(["train", "test", "validation"])
    lab_dir = os.path.join(data_dir, 
                           "labels", 
                           f"{mode}-annotations-human-imagelabels-boxable.csv")
    prefix = "oidv6-" if mode == "train" else ""
    anno_dir = os.path.join(data_dir, 
                            "boxes", 
                            f"{prefix}{mode}-annotations-bbox.csv")
    
    img_to_label = {}
    with open(lab_dir, newline="") as csvfile:
        for ri, row in enumerate(csv.reader(csvfile, delimiter=' ', quotechar='|')):
            if ri == 0: continue

            assert len(row) == 1
            im_id, ver, cno, conf = tuple(row[0].split(","))            
            cno = class_names[cno]

            if im_id not in img_to_label: 
                img_to_label[im_id] = {}
                
            if cno not in img_to_label[im_id]:
                img_to_label[im_id][cno] = {'ver': [], 'conf': []}
            img_to_label[im_id][cno]['ver'].append(ver)
            img_to_label[im_id][cno]['conf'].append(conf)
        
        
    for im_id in img_to_label:
        for lab in img_to_label[im_id]:
            assert len(np.unique(img_to_label[im_id][lab]['conf'])) == 1
            img_to_label[im_id][lab]['conf'] = img_to_label[im_id][lab]['conf'][0]
            
    with open(anno_dir, newline="") as csvfile:
        for ri, row in enumerate(csv.reader(csvfile, delimiter=' ', quotechar='|')):
            if ri == 0: continue
            assert len(row) == 1
            rs = row[0].split(",")
            im_id, src, cno = tuple(rs[:3])
            cno = class_names[cno]
            
            box = [float(v) for v in rs[4:8]]
            if 'box' not in img_to_label[im_id][cno] or src == 'activemil': 
                img_to_label[im_id][cno]['box'] = box
    
    return img_to_label


def make_dataset(dir, mode, sample_info, 
                 class_to_idx, class_to_idx_comp, extensions):
    
    images = []
    allowed_labels = set(class_to_idx.keys())
    Nclasses = len(set(class_to_idx.values()))
    
    for k, v in sample_info.items():
        
        img_path = os.path.join(dir, "images", mode, k + ".jpg")
        
        pos_labels = set([l for l in v.keys() if v[l]['conf'][0] == '1'])
        neg_labels = set([l for l in v.keys() if v[l]['conf'][0] == '0'])

        pos_labels = pos_labels.intersection(allowed_labels)
        neg_labels = neg_labels.intersection(allowed_labels)
        if Nclasses == 601 or len(pos_labels) != 0:
            label = [0] * Nclasses
            all_labels = [0] * 601
            for p in pos_labels:
                label[class_to_idx[p]] = 1
                all_labels[class_to_idx_comp[p]] = 1
            for n in neg_labels:
                if label[class_to_idx[n]] == 0:
                    label[class_to_idx[n]] = -1
                all_labels[class_to_idx_comp[n]] = -1
            item = (img_path, label, all_labels)
            images.append(item)
        
    return images

class OIDatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, train=True, extensions=IMG_EXTENSIONS, 
                 loader=default_loader, transform=None,
                 target_transform=target_transform_oi, label_mapping=None,
                 download=False):
        classes, class_to_idx, code_to_class = self._find_classes(root)
        class_to_idx_comp = {k: v for k, v in class_to_idx.items()}
        if label_mapping is not None:
            classes, class_to_idx = label_mapping(classes, class_to_idx)
        
        mode = "train" if train else "test"
        sample_info = get_image_annotations_mode(code_to_class,
                                                 mode=mode,
                                                 data_dir=root)
    
    
        samples = make_dataset(root, mode, sample_info, 
                                class_to_idx, class_to_idx_comp,
                                extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.all_targets = [s[2] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        """
        code_to_class = load_class_desc(dir)
        classes = [v for v in code_to_class.values()]
        class_to_idx = {code_to_class[k]: i for i, k in enumerate(code_to_class)}
        return classes, class_to_idx, code_to_class

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, comp_target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.target_transform is not None:
            comp_target = self.target_transform(comp_target)
        return sample, target, comp_target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def get_label_map(data_dir):
    CLASS_NAMES = load_class_desc(data_dir)
    label_map = {i: v for i, v in enumerate(CLASS_NAMES.values())}
    return label_map

def get_labels(targ, label_map):
    pos_labels, neg_labels = [], []
    for ti, t in enumerate(targ.numpy()):
        if t == 1:
            pos_labels.append(f"+ {label_map[ti]}")
        elif t == -1:
            neg_labels.append(f"- {label_map[ti]}")
    return ", ".join(pos_labels) + " | " + ", ".join(neg_labels)
