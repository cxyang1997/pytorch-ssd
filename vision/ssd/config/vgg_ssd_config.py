import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 224
image_mean = np.array([0.485, 0.456, 0.406])  # RGB layout
image_std = np.array([0.229, 0.224, 0.225])

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
    SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
    SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
]


priors = generate_ssd_priors(specs, image_size)
