from tensorflow.keras.callbacks import Callback

class Validation(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        """
            For every image in the generator self.generator, calculate the IOU and MSE

            Set the total IOU value:
                logs["val_iou"] = iou

            Set the total MSE:
                logs["val_mse"] = mse
        """
        raise NotImplemented
