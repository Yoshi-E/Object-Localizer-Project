import math
from tensorflow.keras.utils import Sequence

class ImageSequence(Sequence):
    #  rescale=1./255,
    #   rotation_range=30,
    #   width_shift_range=0.2,
    #   height_shift_range=0.2,
    #   shear_range=0.2,
    #   zoom_range=0.2,
    #   horizontal_flip=True,
    #   fill_mode='nearest'
    def __init__(self, img_size=96, batch_size=32):
        self.paths = []
        self.coords = []
        self.IMAGE_SIZE = img_size
        self.BATCH_SIZE = batch_size

    def __len__(self):
        return math.ceil(len(self.coords) / self.BATCH_SIZE)

    def __getitem__(self, idx):
        """
        
        return batch_images, batch_coords
            batch_images is a numpy list of preprocessed images
            batch_coords is a list of the boundingbox cords
        """
        raise NotImplemented
