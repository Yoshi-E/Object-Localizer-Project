import glob
import sys
import os
import xml.etree.ElementTree as ET
from lxml import etree
import cv2
import csv
import pathlib
import random

# importing sibiling package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from  models.Core.config import Config

# Generate a dataset
class DataSetGenerator():
    def __init__(self, dataset_path, skip=1, limit=None, split=0.8):
        if(isinstance(dataset_path, str)):
            self.DATASET_FOLDERS = [dataset_path]
        else:
            self.DATASET_FOLDERS = dataset_path
        self.SPLIT = split
        self.SKIP = skip
        self.LIMIT = limit 

    def clamp(self, n, minn=0, maxn=0):
        return max(min(maxn, n), minn)

    def getXML(self):
        class_names = {}
        k = 0
        output = []
        xml_files = []
        for dataset_path in self.DATASET_FOLDERS:
            if not os.path.exists(dataset_path):
                raise NotADirectoryError("Dataset '{}' not found".format(dataset_path))
            xml_files += glob.glob("{}/*xml".format(dataset_path))

        for i, xml_file in enumerate(xml_files):
            if i % self.SKIP != 0:  # skip every xth image (usefull for videos to prevent overfitting)
                continue
            if len(output) > self.LIMIT:  # limit the dataset size 
                break

            tree = ET.parse(xml_file)

            path = os.path.join(dataset_path, tree.findtext("./filename"))

            height = int(tree.findtext("./size/height"))
            width = int(tree.findtext("./size/width"))
            xmin = int(tree.findtext("./object/bndbox/xmin"))
            ymin = int(tree.findtext("./object/bndbox/ymin"))
            xmax = int(tree.findtext("./object/bndbox/xmax"))
            ymax = int(tree.findtext("./object/bndbox/ymax"))
            try:
                rotation = int(tree.findtext("./object/bndbox/rotation"))
            except:
                rotation = -1
            basename = os.path.basename(path)
            basename = os.path.splitext(basename)[0]
            class_name = basename[:basename.rfind("_")].lower()
            if class_name not in class_names:
                class_names[class_name] = k
                k += 1

            #validate bounding box
            if xmin >= xmax:
                xmin = xmax-1
            if ymin >= ymax:
                ymin = ymax-1
            if xmax > width:
                xmax = width
            if ymax > height:
                ymax = height
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0

            output.append({ "path": path,
                            "height": height,
                            "width": width,
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                            "rotation": rotation,
                            "class_name": class_name,
                            "class_names": class_names[class_name]
                        })

        #output.sort(key=lambda tup : tup[-1])
        output = sorted(output, key=lambda k: k['class_names'])
        return output
    
    def get_train_validate(self, data):
        
        lengths = []
        i = 0
        last = 0
        for j, row in enumerate(data):
            if last == int(row["class_names"]):
                i += 1
            else:
                #print("class {}: {} images".format(data[j-1][-2], i))
                lengths.append(i)
                i = 1
                last += 1

        #print("class {}: {} images".format(data[j-1][-2], i))
        lengths.append(i)

        rows_train = []
        rows_validate = []
        s = 0
        for c in lengths:
            for i in range(c):
                print("{}/{}".format(s + 1, sum(lengths)), end="\r")
                row_dic = data[s]
                if i <= c * self.SPLIT:
                    rows_train.append(row_dic)
                else:
                    rows_validate.append(row_dic)
                s += 1
        return [rows_train, rows_validate]     

class txtDataSetGenerator(DataSetGenerator):

    def __init__(self, dataset_path, skip=1, limit=None, split=0.8):
        super().__init__(dataset_path, skip=skip, limit=limit, split=split)
        self.cfg = Config()

    def validateBounds(self, img, x0, y0, x1, y1):
        height, width, channels = img.shape

        #validate bounding box
        x0 = int(self.clamp(x0*width, 0, width))
        x1 = int(self.clamp(x1*width, 0, width))
        y0 = int(self.clamp(y0*height, 0, height))
        y1 = int(self.clamp(y1*height, 0, height))

        return x0, y0, x1, y1

    def getTXTsData(self, randomize=False):
        class_names = {}
        k = 0
        output = []
        txt_files = []
        for dataset_path in self.DATASET_FOLDERS:
            if not os.path.exists(dataset_path):
                raise NotADirectoryError("Dataset '{}' not found".format(dataset_path))
            txt_files += glob.glob("{}/*.txt".format(dataset_path))

        for i, txt_file in enumerate(txt_files):
            if i % self.SKIP != 0:  # skip every xth image (usefull for videos to prevent overfitting)
                continue
            if len(output) > self.LIMIT:  # limit the dataset size 
                break

            with open(txt_file, "r") as file:
                row = file.readline()
            try:
                class_name = row.split()[0]
                x, y, w, h = [ float(i) for i in row.split()[1:]]
            except (ValueError, IndexError):
                continue

            #if class_name != "0":
            #    continue

            if class_name not in class_names:
                class_names[class_name] = k
                k += 1
            
            # convert yolo format to standard bounding box
            x0 = x-w/2
            x1 = x+w/2
            y0 = y-h/2
            y1 = y+h/2

            imgf = txt_file.replace(".txt", ".jpg")
            path = os.path.relpath(imgf, start=self.cfg.ROOT)
            p = pathlib.Path(path)
            path = str(pathlib.Path(*p.parts[2:])).replace("\\", "/")


            img = cv2.imread(imgf)
            x0, y0, x1, y1 = self.validateBounds(img, x0, y0, x1, y1)
            height, width, channels = img.shape
            output.append({ "path": path,
                            "height": height,
                            "width": width,
                            "xmin": x0,
                            "ymin": y0,
                            "xmax": x1,
                            "ymax": y1,
                            "class_name": class_name,
                            "class_names": class_names[class_name]
                        })
        if randomize:
            random.shuffle(output)
            
        with open(self.cfg.TRAIN_CSV, "w", newline='') as train, open(self.cfg.VALIDATION_CSV, "w", newline='') as validate:
            writer = csv.writer(train, delimiter=",")
            writer2 = csv.writer(validate, delimiter=",")

            c = 0
            SKIP = 5
            for i, row in enumerate(output):
                if c > SKIP:
                    writer2.writerow(row.values())
                    c = 0
                else:
                    writer.writerow(row.values())
                c += 1
                 

        print("\nDone!")
