
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import epsilon
import cv2
import sys
import os
import numpy as np
# importing sibiling package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Core

from .validation import Validation
from .imageSequence import ImageSequence

class FastModel(Core.FastModel):
    def __init__(self, config: Core.Config):
        super().__init__(config)
        self.model = None
        self.__name__ = "MobiNetV2"

    def get_model(self, trainable=False):
        """
        Creates / Loades a Keras model, and returns a Keras Model
        
        
        return Model
        """
        model = MobileNetV2(input_shape=(self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE, 3), include_top=False, alpha=self.cfg.ALPHA)

        # freeze layers
        for layer in model.layers:
            layer.trainable = trainable

        x = model.layers[-1].output
        x = Conv2D(4, kernel_size=3, name="coords_out")(x)
        x = Reshape((4,))(x)
        model.summary()

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
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=[])
        MODE = "max"
        METRIC = "val_iou"
        train_datagen = ImageSequence(self.cfg, self.cfg.TRAIN_CSV)
        validation_datagen = Validation(generator=ImageSequence(self.cfg, self.cfg.VALIDATION_CSV))
        reduce_lr = ReduceLROnPlateau(monitor=METRIC, factor=0.2, patience=8, min_lr=1e-7, verbose=1, mode=MODE)
        stop = EarlyStopping(monitor=METRIC, patience=self.cfg.PATIENCE, mode=MODE)
        checkpoint = ModelCheckpoint("weight-{val_iou:.2f}.h5", monitor=METRIC, verbose=1, save_best_only=True,
                                    save_weights_only=True, mode=MODE)
        
        model.fit_generator(generator=train_datagen,
                            epochs=self.cfg.EPOCHS,
                            callbacks=[validation_datagen, checkpoint, reduce_lr, stop],
                            workers=self.cfg.THREADS,
                            use_multiprocessing=self.cfg.MULTI_PROCESSING,
                            shuffle=True,
                            verbose=1)


    def image(self, image):
        if type(image) == str:
            image = cv2.imread(image)
        if not self.model:
            self.model = self.get_model()
            self.model.load_weights(self.cfg.WEIGHTS_FILE)
        try:
            image_height, image_width, _ = image.shape
            image = cv2.resize(image, (self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE))
            feat_scaled = preprocess_input(np.array(image, dtype=np.float32))


            region = self.model.predict(x=np.array([feat_scaled]))[0]

            x0 = int(region[0] * image_width / self.cfg.IMAGE_SIZE)
            y0 = int(region[1] * image_height / self.cfg.IMAGE_SIZE)

            x1 = int((region[0] + region[2]) * image_width / self.cfg.IMAGE_SIZE)
            y1 = int((region[1] + region[3]) * image_height / self.cfg.IMAGE_SIZE)
            
            return (x0, y0, x1, y1)
        except Exception as e:
            print("Could not process image '{}'".format(e))


        