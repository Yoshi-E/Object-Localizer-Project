from models import VGG16
from models.Core import Config
import os
from glob import glob
config = Config()
model = VGG16.FastModel(config)
config.WEIGHTS_FILE = "weights/VGG16/weight-0.62.h5"


if __name__ == "__main__":
    #model.test_image("FastDetector/datasets/10_rosbag/images/1565608339175915704.jpg")
    model.test_images(glob("FastDetector/datasets/10_rosbag/images/*.jpg"), skip=100)