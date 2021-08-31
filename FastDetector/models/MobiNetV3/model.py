from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.applications import MobileNetV3Small
#from tensorflow.keras.mobilenet_v3 import MobileNetV3, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Conv2D, Reshape, Multiply
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import epsilon
import cv2
import os
import numpy as np
import sys

# importing sibiling package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Core

# inherit the pervious model and functions
import MobiNetV2
from MobiNetV2.validation import Validation
from MobiNetV2.imageSequence import ImageSequence
 


class FastModel(MobiNetV2.FastModel):
    def __init__(self, config: Core.Config):
        super().__init__(config)
        self.model = None
        self.__name__ = "MobiNetV3"

    def get_model(self, trainable=False):
        """
        Creates / Loades a Keras model, and returns a Keras Model
        
        
        return Model
        """
        model = MobileNetV3Small(input_shape=(self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE, 3), include_top=False, alpha=self.cfg.ALPHA)

        # freeze layers
        for layer in model.layers:
            layer.trainable = trainable

        model.summary()
        print(model.layers[-9])
        x = model.layers[-9].output
        x = Conv2D(4, kernel_size=3, name="coords_out")(x)
        x = Reshape((4,))(x)

        return Model(inputs=model.input, outputs=x)

    def train(self):
        """
        Create and Load the model,
        Load the ImageSequence
        and run the fit_generator on the model

        returns None
        """
        model = self.get_model()
        model.summary()

        train_datagen = ImageSequence(self.cfg, self.cfg.TRAIN_CSV)
        validation_datagen = Validation(generator=ImageSequence(self.cfg, self.cfg.VALIDATION_CSV))

        model.compile(loss="mean_squared_error", optimizer="adam", metrics=[])

        checkpoint = ModelCheckpoint("model-{val_iou:.2f}.h5", monitor="val_iou", verbose=1, save_best_only=True,
                                    save_weights_only=True, mode="max")
        stop = EarlyStopping(monitor="val_iou", patience=self.cfg.PATIENCE, mode="max")
        reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.2, patience=8, min_lr=1e-7, verbose=1, mode="max")

        model.fit_generator(generator=train_datagen,
                            epochs=self.cfg.EPOCHS,
                            callbacks=[validation_datagen, checkpoint, reduce_lr, stop],
                            workers=self.cfg.THREADS,
                            use_multiprocessing=self.cfg.MULTI_PROCESSING,
                            shuffle=True,
                            verbose=1)

        