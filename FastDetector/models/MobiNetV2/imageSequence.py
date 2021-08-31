import math

from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.utils import Sequence
from keras.preprocessing import image

import sys
import os
# importing sibiling package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Core
import numpy as np
import csv
import math
from PIL import Image

class ImageSequence(Core.ImageSequence):
    def __init__(self, config: Core.Config, csv_file):
        self.paths = []
        self.coords = None
        self.IMAGE_SIZE = config.IMAGE_SIZE
        self.BATCH_SIZE = config.BATCH_SIZE
        self.database_root = os.path.join(config.ROOT, config.DATASET)

        with open(csv_file, "r") as file:
            self.coords = np.zeros((sum(1 for line in file), 4))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")
            for i, row in enumerate(reader):
                for ii, r in enumerate(row[1:7]):
                    row[ii+1] = int(r)

                path, image_height, image_width, x0, y0, x1, y1, _, _ = row
                self.coords[i, 0] = x0 * self.IMAGE_SIZE / image_width
                self.coords[i, 1] = y0 * self.IMAGE_SIZE / image_height
                self.coords[i, 2] = (x1 - x0) * self.IMAGE_SIZE / image_width
                self.coords[i, 3] = (y1 - y0) * self.IMAGE_SIZE / image_height 

                self.paths.append(os.path.join(self.database_root, path))

    def __len__(self):
        return math.ceil(len(self.coords) / self.BATCH_SIZE)

    def __getitem__(self, idx):
        """
        
        return batch_images, batch_coords
            batch_images is a numpy list of preprocessed images
            batch_coords is a list of the boundingbox cords
        """
        batch_paths = self.paths[idx * self.BATCH_SIZE:(idx + 1) * self.BATCH_SIZE]
        batch_coords = self.coords[idx * self.BATCH_SIZE:(idx + 1) * self.BATCH_SIZE]

        batch_images = np.zeros((len(batch_paths), self.IMAGE_SIZE, self.IMAGE_SIZE, 3), dtype=np.float32)
        for i, f in enumerate(batch_paths):
            try:
                # img = image.load_img(f, target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE))
                # x = image.img_to_array(img)
                # #x = np.expand_dims(x, axis=0)
                # x = preprocess_input(x)
                img = Image.open(f)
                img = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
                img = img.convert('RGB')
                npimg = np.array(img, dtype=np.float32)
                batch_images[i] = preprocess_input(npimg)
                img.close()
            except Exception as e:
                print("Failed to process '{}', <{}>".format(f, e))

        return batch_images, batch_coords
