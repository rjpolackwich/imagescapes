import boto3
import coco
import os.path
from PIL import Image
import tempfile
from torch.utils.data import Dataset
from torchvision import transforms


class CocoClassificationDataset(Dataset):
    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, coco_file, transform=None, max_count=0, session=None):
        self.dataset = coco.Dataset.from_file(coco_file)
        self.image_ids = sorted(self.dataset.images.keys())
        if max_count > 0:
            self.image_ids = self.image_ids[:max_count]
        self.transform = transform or self.default_transform
        self._temp_dir = None

    @property
    def temp_dir(self):
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
        return self._temp_dir

    def load_image(self, image):
        if image.image_path.startswith('file://'):
            image = Image.open(image.image_path[6:])

        if image.image_path.startswith('s3://'):
            local_path = os.path.join(self.temp_dir.name, image.id)
            try:
                image = Image.open(local_path)
            except FileNotFoundError:
                bucket, key = image.image_path[5:].split('/', 1)
                s3 = boto3.client('s3')
                s3.download_file(bucket, key, local_path)
                image = Image.open(local_path)

        return self.transform(image)

    def __getitem__(self, key):
        image_id = self.image_ids[key]
        image = self.dataset.images[image_id]
        annotation_ids = self.dataset.annotations_by_image[image_id]
        if len(annotation_ids) > 1:
            raise ValueError(f'Too many annotations for image `{image_id}`: found `{len(annotation_ids)}` annotations')

        category_id = self.dataset.annotations[annotation_ids[0]].category_id
        loaded_image = self.load_image(image)
        return loaded_image, category_id

    def __len__(self):
        return len(self.image_ids)
