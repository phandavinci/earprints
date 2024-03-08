from ultralytics import YOLO
import os
model = YOLO('yolo.pt')
data = '/home/phanda/Projects/data'
for directory in os.listdir(data):
    results = model(os.path.join(data, directory))
    for result in results:
        result.save_crop(os.path.join(data,'cropped'+directory))
    print("cropped all the images in "+directory)