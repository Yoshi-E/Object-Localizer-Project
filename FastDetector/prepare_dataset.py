from tools.convertLabels import txtDataSetGenerator
from models.Core import Config
import os

config = Config()

DATASET = os.path.join(config.ROOT, "datasets/10_rosbag/images/")

dsg = txtDataSetGenerator(DATASET, limit=9999999)
dsg.getTXTsData(randomize=True)
