import random
data = []


#Simple script used to randomize and merge training sets
files = ["FastDetector/datasets/10_rosbag/train.csv"]

for f in files:
    with open(f,'r') as source:
        #data = [ (random.random(), line) for line in source ]
        for line in source:
            data.append(line)  
random.shuffle(data)
#data.sort()
with open('train.csv','w') as target:
    for line in data:
        #print(line)
        target.write( line )
