from torchvision import transforms

def build_transform(chip_size,
                    rotate=False,
                    rotation_angle=45,
                    flip=False,
                    mean_norm=[0.485, 0.456, 0.406],
                    mean_std=[0.229, 0.224, 0.225],
                    ):
    transform_list = [transforms.Resize(chip_size)]
    if rotate:
        transform_list.append(transforms.RandomRotation(rotation_angle))
    if flip:
        transform_list.append(transforms.RandomVerticalFlip())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean_norm, mean_std))

    return transforms.Compose(transform_list)



