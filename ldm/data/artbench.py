from torchvision.datasets import ImageFolder
import numpy as np
import albumentations
import PIL
from PIL import Image
from torchvision import transforms
class PrepDataset(ImageFolder):
    def __init__(self, root, size=None, interpolation="bicubic", flip_p=0.5, **kwargs):
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        super().__init__(root)


    def preprocess_image(self, image):
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        
        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        ele, label = super().__getitem__(i)
        example["image"] = self.preprocess_image(ele)
        example["class_label"] = label
        return example


class PrepDatasetCond(ImageFolder):
    def __init__(self, root, size=None, random_crop=False, labels=None, **kwargs):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

        super().__init__(root)



    def preprocess_image(self, image):
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        ele, label = super().__getitem__(i)
        #print(ele)
        example["image"] = self.preprocess_image(ele)
        example["class_label"] = label
        example["human_label"] = label
        return example

class ArtBench10Train(PrepDatasetCond):
    def __init__(self, **kwargs):
        super().__init__(root="data/artbench/train", **kwargs)


class ArtBench10Validation(PrepDatasetCond):
    def __init__(self, **kwargs):
        super().__init__(root="data/artbench/validation", **kwargs)


class ArtBench10Impressionism(PrepDataset):
    def __init__(self, **kwargs):
        super().__init__(root="data/artbench-impressionism", **kwargs)