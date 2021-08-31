import os
import sys
import glob
import cv2
from lxml import etree

# A collection of functions and tools used in the menu 
class Tools():
    @staticmethod
    def query_yes_no(question, default="yes"):
        #taken from http://code.activestate.com/recipes/577058/
        valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                 "(or 'y' or 'n').\n")
    
    @staticmethod
    def column(matrix, i):
        return [row[i] for row in matrix]

    @staticmethod
    def viewImagesBox(file):
        file = file.replace('"', "")
        if(os.path.exists("../blender/Renders/"+file)):
            raise Exception("Invalid set")
        for filename in glob.glob("../blender/Renders/" + file + "/*png"):
            Tools.viewImageBox(filename)
        for filename in glob.glob("../blender/Renders/" + file + "/*jpg"):
            Tools.viewImageBox(filename)

    @staticmethod
    def viewImageBox(filename):
            filename = filename.replace('"', "")
            xml = filename.replace(".png", ".xml")
            xml = xml.replace(".jpg", ".xml")

            tree = etree.parse(xml)
            root = tree.getroot()
            xmin = int(root[5][4][0].text)
            ymin = int(root[5][4][1].text)
            xmax = int(root[5][4][2].text)
            ymax = int(root[5][4][3].text)

            unscaled = cv2.imread(filename)

            print("{}: ({},{})({},{}) {}°".format(filename, xmin, ymin, xmax, ymax))
            cv2.rectangle(unscaled, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.imshow("{} ".format(filename), unscaled)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    @staticmethod
    def predict_dataset(custommodel, WEIGHTS_FILE, DATASET, save=False):
        IMAGE_SIZE = custommodel.IMAGE_SIZE

        data_path = "datasets/{}/".format(DATASET)
        if not os.path.exists(data_path):
            raise NotADirectoryError("Dataset '{}' not found".format(data_path))
        
        model = custommodel.create_model()
        model.load_weights(WEIGHTS_FILE)
        for filename in glob.glob(data_path+"*xml"):
            png = filename.replace(".xml", ".png")
            if(os.path.isfile(png)==False):
                png = png.replace(".png", ".jpg")
            
            tree = etree.parse(filename)
            root = tree.getroot()
            xmin = int(root[5][4][0].text)
            ymin = int(root[5][4][1].text)
            xmax = int(root[5][4][2].text)
            ymax = int(root[5][4][3].text)

            unscaled = cv2.imread(png)
            image_height, image_width, _ = unscaled.shape


            regions = custommodel.predict(model, unscaled) 
            cv2.rectangle(unscaled, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            for region in regions:
                x0 = int(region[0] * image_width / IMAGE_SIZE)
                y0 = int(region[1] * image_height / IMAGE_SIZE)

                x1 = int((region[0] + region[2]) * image_width / IMAGE_SIZE)
                y1 = int((region[1] + region[3]) * image_height / IMAGE_SIZE)
                if(save==False):
                    print("{}: ({},{})({},{}) {}°".format(filename, xmin, ymin, xmax, ymax))
                cv2.rectangle(unscaled, (x0, y0), (x1, y1), (0, 0, 255), 1)

            if(save==False):
                cv2.imshow("{} rotation: {}".format(filename), unscaled)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print("#"*50)
            else:
                raise NotImplementedError()
