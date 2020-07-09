from collections import defaultdict
from dataclasses import dataclass
import dateparser
from datetime import datetime
from shapely.geometry import Polygon
from typing import Any, Dict, List, Optional, Union
import ujson as json


@dataclass
class Info:
    version: str
    contributor: str
    date_created: datetime
    description: Optional[str] = None
    s3_path: Optional[str] = None

    @classmethod
    def from_json(cls, data):
        data['date_created'] = dateparser.parse(data['date_created'])
        return cls(**data)


@dataclass
class License:
    id: int
    name: str
    url: str

    @classmethod
    def from_json(cls, data):
        return cls(**data)


@dataclass
class Image:
    id: int
    width: int
    height: int
    image_path: str
    epsg_code: Optional[int] = None
    image_bounds: Optional[Polygon] = None
    date_captured: Optional[datetime] = None

    @classmethod
    def from_json(cls, data):
        if 'src_crs' in data:
            data['epsg_code'] = int(data.pop('src_crs'))

        if 'date_captured' in data:
            data['date_captured'] = dateparser.parse(date['date_captured']) 
        return cls(**data)


@dataclass
class SegmentationMask:
    rle = None
    polygon = None

    def __init__(self, mask, type):
        if type == 'rle':
            self.type = 'rle'
            self.rle = mask
            return

        if type == 'polygon':
            self.type = 'polygon'
            self.polygon = mask
            return

        raise ValueError(f'Invalid segmentation mask type `{type}`')

    @classmethod
    def from_rle(cls, rle):
        return cls(rle, 'rle')

    @classmethod
    def from_polygon(cls, polygon):
        return cls(polygon, 'polygon')

    @classmethod
    def from_json(cls, data):
        if isinstance(data, list):
            return cls.from_polygon(data)

        return cls.from_rle(data)


@dataclass
class InstanceAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: SegmentationMask
    area: float
    bbox: Polygon
    iscrowd: bool

    @classmethod
    def from_json(cls, data):
        data['segmentation'] = SegmentationMask.from_json(data['segmentation'])
        x_min, y_min, width, height = data['bbox']
        data['bbox'] = box(x_min, y_min, x_min + width, y_min + height)
        return cls(**data)


@dataclass
class ClassificationAnnotation:
    id: int
    image_id: int
    category_id: int

    @classmethod
    def from_json(cls, data):
        return cls(**data)


Annotation = Union[InstanceAnnotation, ClassificationAnnotation]


@dataclass
class Category:
    id: int
    name: str
    supercategory: Optional[str] = None

    @classmethod
    def from_json(cls, data):
        return cls(**data)


class Dataset:
    info: Dict[str, Any]
    annotations: Dict[int, Annotation]
    images: List[Image]
    categories: List[Category]

    def __init__(self, info=None, annotations=[], images=[], categories=[]):
        self.info = info or {}
        self.categories = {category.id: category for category in categories}
        self.images = {image.id: image for image in images}

        self.annotations = {}
        self.annotations_by_image = defaultdict(list)
        self.images_by_category = defaultdict(list)
        for annotation in annotations:
            self.annotations[annotation.id] = annotation
            self.annotations_by_image[annotation.image_id].append(annotation.id)
            if annotation.image_id not in self.images_by_category[annotation.category_id]:
                self.images_by_category[annotation.category_id].append(annotation.image_id)

    @classmethod
    def from_json(cls, data):
        if 'licenses' in data:
            del data['licenses']

        info = Info.from_json(data['info'])
        annotation_cls = cls.get_annotation_class(data)
        annotations = [annotation_cls.from_json(annotation) for annotation in data['annotations']]
        images = [Image.from_json(image) for image in data['images']]
        categories = [Category.from_json(category) for category in data['categories']]
        return cls(
            info=info,
            annotations=annotations,
            images=images,
            categories=categories,
        )

    @classmethod
    def from_file(cls, path):
        with open(path) as fh:
            data = json.load(fh)

        return cls.from_json(data)

    @staticmethod
    def get_annotation_class(data):
        sample_annotation = data['annotations'][0]
        if 'bbox' in sample_annotation or 'segmentation' in sample_annotation:
            return InstanceAnnotation

        return ClassificationAnnotation
