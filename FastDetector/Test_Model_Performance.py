from MobiNetV2_test import model as mobiNetV2
from MobiNetV3_test import model as mobiNetV3
from VGG16_test import model as vgg16
#from CustomV1_test import model as customV1
from models.Core import Config
import os
from glob import glob

# we select the first 10000 images, by taking every 1th image.
test_images = list(glob("FastDetector/datasets/10_rosbag/images/*.jpg"))[:10000:1]

results = ""
for model in [mobiNetV2, mobiNetV3, vgg16]:
    total_time = 0
    lows = 9999
    highs = 0

    model.image(test_images[0]) #testing an images forces the model to load
    for image in test_images:
        delta = model.time_image(image)
        if delta < lows:
            lows = delta
        if delta > highs:
            highs = delta
        total_time += delta


    results += "{:<10}: TOTAL: {:.2f}s LOW: {:.5f} HIGH: {:.5f}\n".format(model.__name__, total_time, lows, highs)
print(results)
