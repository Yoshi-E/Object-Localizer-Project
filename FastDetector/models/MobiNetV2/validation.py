from tensorflow.keras.callbacks import  Callback
from tensorflow.keras.backend import epsilon
from tensorflow.keras.metrics import MeanIoU
import numpy as np

import sys
import os
# importing sibiling package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Core

class Validation(Core.Validation):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end_old(self, epoch, logs):
        miou = MeanIoU(num_classes=2)

        for i in range(len(self.generator)):
            images, cords = self.generator[i] 
            pred = self.model.predict_on_batch(images)
            
            #ensure all values are positve
            pred = pred.clip(min=0)
            miou.update_state(cords, pred)

        iou = miou.result().numpy()
        logs["val_iou"] = iou

        print("val_iou: {}".format(iou))

    def on_epoch_end(self, epoch, logs):
        intersections = 0
        unions = 0
        err = 0

        for i in range(len(self.generator)):
            # original images & their ground truth bounding boxes
            # shape: BATCH_SIZE x 4 (x0,y0,x1,y1)
            images, cords = self.generator[i] 

            # predictiing all batch images
            pred = self.model.predict_on_batch(images)

            # calculate the diffrence (pred, truth) for all coordinates and take the avrage  
            err += np.linalg.norm(cords - pred, ord='fro') / len(pred)

            # set all negative values to 0
            pred = pred.clip(min=0)

            # the difference between the truth and the prediction is calculated efficiently
            diff_width = np.minimum(cords[:,0] + cords[:,2], pred[:,0] + pred[:,2]) - np.maximum(cords[:,0], pred[:,0])
            diff_height = np.minimum(cords[:,1] + cords[:,3], pred[:,1] + pred[:,3]) - np.maximum(cords[:,1], pred[:,1])
            intersection = diff_width.clip(min=0) * diff_height.clip(min=0)

            # For the IOU first both areas of the predictions are calculated
            # IOU = true_positive / (true_positive + false_positive + false_negative)
            area_truth = np.abs(cords[:,0] - cords[:,2]) * np.abs(cords[:,1] - cords[:,3])
            area_pred = np.abs(pred[:,0] - pred[:,2]) * np.abs(pred[:,1] - pred[:,3])
            union = area_truth + area_pred - intersection
            union = union.clip(min=0)

            intersections += np.sum(intersection * (union > 0))
            unions += np.sum(union)

        # We calculate again the avrg iou over all images
        # We add epsilon to prevent division by zero
        iou = np.round(intersections / (unions + epsilon()), 4)
        logs["val_iou"] = iou

        # Err represents the combined error of all the cooridinates x0, y0, x1, y1 for the truth and prediction 
        err = np.round(err, 4)
        logs["val_err"] = err

        print(f"val_iou: {iou} - val_err: {err}")