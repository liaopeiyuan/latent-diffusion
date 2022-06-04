from torchvision.datasets import ImageFolder


class ArtBench10Train(ImageFolder):
    def __init__(self, **kwargs):
        super().__init__(root="data/artbench/train", **kwargs)


class ArtBench10Validation(ImageFolder):
    def __init__(self, **kwargs):
        super().__init__(root="data/artbench/validation", **kwargs)
