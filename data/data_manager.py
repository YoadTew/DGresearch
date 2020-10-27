from torch.utils import data
from data.PacsHDF5Dataset import HDF5Dataset


def ds_name_to_path(data_dir, name, mode='train'):
    return f'{data_dir}\\{name}_{mode}.hdf5'


def get_train_loader(args):
    source_dataset_names = [ds_name_to_path(args.data_dir, x, 'train') for x in args.source]

    source_datasets = []
    for dname in source_dataset_names:
        dataset = HDF5Dataset(dname, args.image_size, mode='train')
        source_datasets.append(dataset)

    train_dataloader = data.DataLoader(data.ConcatDataset(source_datasets), num_workers=1,
                                       batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_dataloader


def get_val_loader(args):
    val_dataset_names = [ds_name_to_path(args.data_dir, x, 'val') for x in args.source]

    val_datasets = []
    for dname in val_dataset_names:
        dataset = HDF5Dataset(dname, args.image_size, mode='test')
        val_datasets.append(dataset)

    val_dataloader = data.DataLoader(data.ConcatDataset(val_datasets), num_workers=1,
                                       batch_size=args.batch_size, shuffle=True, drop_last=True)

    return val_dataloader


def get_test_loader(args):

    dname = ds_name_to_path(args.data_dir, args.target, 'test')
    dataset = HDF5Dataset(dname, args.image_size, mode='test')

    test_dataloader = data.DataLoader(dataset, num_workers=1,
                                       batch_size=args.batch_size, shuffle=True, drop_last=True)

    return test_dataloader
