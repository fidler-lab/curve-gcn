import json
from torch.utils.data import DataLoader
from Dataloaders import cityscapes_processed_hard

def get_data_loaders():
    opts = json.load(open('Experiments/gnn-active-spline.json', 'r'))
    dataset_train = cityscapes_processed_hard.DataProvider(split='train', opts=opts['dataset']['train'])
    dataset_val = cityscapes_processed_hard.DataProvider(split='train_val', opts=opts['dataset']['train_val'])

    train_loader = DataLoader(dataset_train, batch_size=16,
        shuffle=True, num_workers=4, collate_fn=cityscapes_processed_hard.collate_fn, drop_last=True)

    val_loader = DataLoader(dataset_val, batch_size=16,
        shuffle=False, num_workers=4, collate_fn=cityscapes_processed_hard.collate_fn, drop_last=True)

    return train_loader, val_loader


