from models import MobiNetV2
from models.Core import Config
import os

config = Config()
config.DATASET = "datasets/10_rosbag"
config.BATCH_SIZE = 128
config.TRAIN_CSV = os.path.join(config.ROOT, "datasets/10_rosbag/all_train.csv")
config.VALIDATION_CSV = os.path.join(config.ROOT, "datasets/10_rosbag/front_validation.csv")

model = MobiNetV2.FastModel(config)
model.train() 