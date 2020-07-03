from .. import datasets
from .constants import DATASET_PATHS
import torch as ch
import pytest
from itertools import combinations_with_replacement as cwr
from torchvision import transforms

"""
Tests:
    - Make sure we can load all of the datasets and make working loaders
    - Make sure that subset and the related arguments work
    - Turn data augmentation on/off 
    - Make sure transforms are settable
    - Make sure shuffling works as expected
    - Make sure Custom ImageNet hierarchy is working
"""


@pytest.fixture(scope="session")
def standard_datasets():
    """
    Setup script, loads datasets with default options
    """
    std_datasets= []
    assert datasets.DATASETS.keys() == DATASET_PATHS.keys()
    for ds_name in datasets.DATASETS:
        if DATASET_PATHS[ds_name] is None: continue
        std_datasets.append((datasets.DATASETS[ds_name], 
                            DATASET_PATHS[ds_name]))
    return std_datasets


class TestInit:
    def test_load_datasets_no_error(self, standard_datasets):
        """
        Check that we can load all of the datasets 
            with the proper batch size
        """
        for ds, path in standard_datasets:
            ds = ds(path)
            tl, vl = ds.make_loaders(batch_size=50, workers=10)
            train_batch = next(iter(tl))
            test_batch = next(iter(vl))
            assert train_batch[0].shape[0] == 50
            assert train_batch[1].shape[0] == 50
            assert test_batch[0].shape[0] == 50
            assert test_batch[1].shape[0] == 50

    def test_initialize_with_arguments(self, standard_datasets):
        for ds_class, path in standard_datasets:
            ds = ds_class(path, mean=ch.zeros(3).float())
            assert ch.all(ds.mean == ch.zeros(3).float())

            ds = ds_class(path, std=ch.ones(1).float())
            assert ch.all(ds.std == ch.ones(1).float())

            with pytest.raises(ValueError):
                ds_class(path, mean=[0, 0, 0])

            with pytest.raises(ValueError):
                ds_class(path, invalid_arg=True)

class TestArguments:
    def test_shuffle(self, standard_datasets):
        """
        Check train and val shuffling (all combinations of train shuffle and
        validation set shuffle)
        """
        for ds, path in standard_datasets:
            ds = ds(path)
            for st, sv in cwr([True, False], 2):
                print(f"Checking shuffle_train={st}, shuffle_val={sv}")
                tl, vl = ds.make_loaders(workers=10, batch_size=50, 
                    shuffle_train=st, shuffle_val=sv)
                tl_p, vl_p = ds.make_loaders(workers=10, batch_size=50, 
                    shuffle_train=st, shuffle_val=sv)
                assert st != (ch.all(next(iter(tl))[1] == next(iter(tl_p))[1]))
                assert sv != (ch.all(next(iter(vl))[1] == next(iter(vl_p))[1]))
    
    def test_data_augmentation(self, standard_datasets):
        train_transform = transforms.Compose([
            transforms.Resize(126),
            transforms.CenterCrop(126),
            transforms.ToTensor()
        ])
        val_transform = transforms.Compose([
            transforms.Resize(235),
            transforms.CenterCrop(235),
            transforms.ToTensor()
        ])
        for ds, path in standard_datasets:
            ds = ds(path, transform_train=train_transform,
                          transform_test=val_transform)
            tl, vl = ds.make_loaders(workers=0, batch_size=50)
            assert list(next(iter(tl))[0].shape) == [50, 3, 126, 126]
            assert list(next(iter(vl))[0].shape) == [50, 3, 235, 235]

class TestSubsets:
    def test_fixed_subset_start(self, standard_datasets):
        """
        Test for "first" subsets
        - Make a subset of size 1000 starting from 0
        - Make a subset of size 500 starting from 500
        - last 500 images of the first loader = first 500 of the second
        """
        for ds, path in standard_datasets:
            ds = ds(path)
            tl, _ = ds.make_loaders(workers=10, batch_size=50, subset=1000, 
                subset_type='first', data_aug=False, shuffle_train=False)
            tl_p, _ = ds.make_loaders(workers=10, batch_size=50, subset=500,
                subset_start=500,  subset_type='first', data_aug=False, 
                shuffle_train=False)
            assert (len(tl) == 20) and (len(tl_p) == 10)
            tl, tl_p = iter(tl), iter(tl_p)
            for i in range(10): next(tl)
            x, y = next(tl)
            x_p, y_p = next(tl_p)
            assert (x_p - x).abs().max() < 1e-2
            assert ch.all(y_p == y)

    def test_random_subset_seed(self, standard_datasets):
        """
        Test for setting the seed in random subsets
        - First try setting no seed, subsets should be different
        - Then try setting fixed different seeds, subsets should be different
        - Then try setting fixed same seeds, subsets should be the same
        - Make sure subset start has no bearing
        """
        for ds, path in standard_datasets:
            ds = ds(path)
            tl, _ = ds.make_loaders(workers=0, batch_size=50, subset=50, 
                subset_type='rand', data_aug=False, shuffle_train=False)
            tl_p, _ = ds.make_loaders(workers=0, batch_size=50, subset=50,
                subset_type='rand', data_aug=False, shuffle_train=False)
            assert not ch.all(next(iter(tl))[1] == next(iter(tl_p))[1])

            tl, _ = ds.make_loaders(workers=0, batch_size=50, subset=50, 
                subset_type='rand', data_aug=False, shuffle_train=False,
                subset_seed=0)
            tl_p, _ = ds.make_loaders(workers=0, batch_size=50, subset=50,
                subset_type='rand', data_aug=False, shuffle_train=False, 
                subset_seed=1)
            assert not ch.all(next(iter(tl))[1] == next(iter(tl_p))[1])

            tl, _ = ds.make_loaders(workers=0, batch_size=50, subset=50, 
                subset_type='rand', data_aug=False, shuffle_train=False,
                subset_seed=0)
            tl_p, _ = ds.make_loaders(workers=0, batch_size=50, subset=50,
                subset_type='rand', data_aug=False, shuffle_train=False, 
                subset_seed=0)
            assert ch.all(next(iter(tl))[1] == next(iter(tl_p))[1])


