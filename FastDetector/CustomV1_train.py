from models import CustomV1
from models.Core import Config
import os

config = Config()
config.DATASET = "datasets/10_rosbag"
config.BATCH_SIZE = 128
config.TRAIN_CSV = os.path.join(config.ROOT, "datasets/10_rosbag/front_train.csv")
config.VALIDATION_CSV = os.path.join(config.ROOT, "datasets/10_rosbag/front_validation.csv")

model = CustomV1.FastModel(config)
model.train()