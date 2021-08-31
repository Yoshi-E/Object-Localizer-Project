
import os

class Config:
    def __init__(self):
        # 0.35, 0.5, 0.75, 1.0
        self.ALPHA = 1.0

        # 96, 128, 160, 192, 224
        self.IMAGE_SIZE = 96

        self.EPOCHS = 200
        self.BATCH_SIZE = 32
        self.PATIENCE = 50

        self.MULTI_PROCESSING = False
        self.THREADS = 11

        self.ROOT = os.path.dirname(os.path.realpath(__file__))
        self.ROOT = os.path.abspath(os.path.join(self.ROOT, os.pardir, os.pardir)) #moving up 2 directories to root
        self.DATASET = "datasets/None/"
        self.TRAIN_CSV = "train.csv"
        self.VALIDATION_CSV = "validation.csv"
        self.WEIGHTS = "weights" # where weights files are saved
        self.WEIGHTS_FILE = None  # weight file loaded into the model
        