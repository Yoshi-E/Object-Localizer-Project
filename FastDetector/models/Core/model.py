from .config import Config
import cv2
import time
import sys

class FastModel(object):
    def __init__(self, config: Config):
        self.cfg = config
        self.__name__ = vars(sys.modules[__name__])['__package__']
        
    def get_model():
        """
        Creates / Loades a Keras model, and returns a Keras Model
        
        
        return Model
        """
        raise NotImplemented

    def train(self):
        """
        Create and Load the model,
        Load the ImageSequence
        and run the fit_generator on the model

        returns None
        """
        raise NotImplemented

    def image(self, image):
        """ 
        Tests an image on a given Model

        image   cv2 image to test
        
        """
        raise NotImplemented    
        
    def time_image(self, image):
        """ 
        Mesures the time of an image test.
        Used to mesure performance.
        """
        start = time.time()
        self.image(image)
        return time.time() - start

    def test_image(self, image):
        """ 
        Tests an image on a given Model

        image   cv2 image to test
        
        """
        if type(image) == str:
            image = cv2.imread(image)
        x0, y0, x1, y1 = self.image(image)

        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def test_images(self, images, skip=100):
        for i, image in enumerate(images):
            if i % skip != 0:
                continue
            if type(image) == str:
                image = cv2.imread(image)
            x0, y0, x1, y1 = self.image(image)

            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1)
            cv2.imshow("image", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()