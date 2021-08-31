from models import CustomV1
from models.Core import Config
import os
from glob import glob
config = Config()
model = CustomV1.FastModel(config)
config.WEIGHTS_FILE = "weights/CustomV1/weight-0.00.h5"

if __name__ == "__main__":
    #model.test_image("FastDetector/datasets/10_rosbag/images/1565608339175915704.jpg")
    model.test_images(glob("FastDetector/datasets/10_rosbag/images/*.jpg"), skip=100)