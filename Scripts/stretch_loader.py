from torch.utils.data import DataLoader
from Dataloaders import cityscapes_processed_stretch
from Dataloaders import custom_transforms as tr
from torchvision import transforms

def get_data_loader():
    composed_transforms_tr = transforms.Compose([
        tr.CropFromMaskStretchMulticomp(crop_elems=('image', 'gt'), relax=10, zero_pad=True),
        tr.FixedResizeStretchMulticomp(resolutions={'crop_image': (224, 224), 'crop_gt': (224, 224)}),
        tr.ToTensorStretchMulticomp()])

    composed_transforms_ts = transforms.Compose([
        tr.CropFromMaskStretchMulticomp(crop_elems=('image', 'gt'), relax=10, zero_pad=True),
        tr.FixedResizeStretchMulticomp(resolutions={'void_pixels': None, 'gt': None, 'crop_image': (224, 224), 'crop_gt': (224, 224)}),
        tr.ToTensorStretchMulticomp()])

    trainset = cityscapes_processed_stretch.CityScapesProcessedStretchMulticomp(train=True, split='train', transform=composed_transforms_tr)
    valset = cityscapes_processed_stretch.CityScapesProcessedStretchMulticomp(train=False, split='train_val', transform=composed_transforms_ts)

    train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4,
                                       collate_fn=cityscapes_processed_stretch.collate_fn, drop_last=True)
    val_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=4,
                                     collate_fn=cityscapes_processed_stretch.collate_fn, drop_last=True)

    return train_loader, val_loader