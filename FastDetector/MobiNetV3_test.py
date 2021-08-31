from models import MobiNetV3
from models.Core import Config
import os
from glob import glob
config = Config()
model = MobiNetV3.FastModel(config)
config.WEIGHTS_FILE = "weights/MobiNetV3/weight-0.73.h5"


if __name__ == "__main__":
    #model.test_image("FastDetector/datasets/10_rosbag/images/1565608339175915704.jpg")
    model.test_images(glob("FastDetector/datasets/10_rosbag/images/*.jpg"), skip=100)