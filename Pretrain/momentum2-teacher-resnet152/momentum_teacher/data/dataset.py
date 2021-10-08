from torchvision.datasets import ImageFolder


class SSL_Dataset(ImageFolder):
    def __init__(self, transform=None, root="/dev/shm/ILSVRC2012_RAW_PYTORCH/train", target_transform=None, is_valid_file=None):
        super(SSL_Dataset, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )
        # self.image_dir + f"imagenet_{'train' if stage in ('train', 'ft') else 'val'}"

        if transform is not None:
            if isinstance(transform, list) and len(transform) > 1:
                self.transform, self.transform_k = transform
            else:
                self.transform, self.transform_k = transform, None
        else:
            raise ValueError("Transform function missing!")

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        img1 = self.transform(sample)
        if self.transform_k is not None:
            img2 = self.transform_k(sample)
        else:
            img2 = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img1, img2], target


class ImageNet(ImageFolder):
    def __init__(self, train, transform=None):
        root = "/dev/shm/ILSVRC2012_RAW_PYTORCH/{}".format('train' if train else 'val')
        super(ImageNet, self).__init__(root, transform=transform )
        self.transform = transform


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
