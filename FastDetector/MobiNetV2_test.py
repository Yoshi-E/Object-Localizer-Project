from models import MobiNetV2
from models.Core import Config
import os
from glob import glob
config = Config()
model = MobiNetV2.FastModel(config)
config.WEIGHTS_FILE = "weights/MobiNetV2/front_weight-0.12.h5"


if __name__ == "__main__":
    #model.test_image("FastDetector/datasets/pets/images/Abyssinian_2.jpg")
    #model.test_images(glob("FastDetector/datasets/pets/images/*.jpg"), skip=1)

    #model.test_image("FastDetector/datasets/10_rosbag/images/1565608339175915704.jpg")
    model.test_images(glob("FastDetector/datasets/10_rosbag/images/*.jpg"), skip=100)