from ultralytics import YOLO
import os
import json
model = YOLO('yolo.pt')
def forChangingWholeDir(path):
    result = model(path)
    result.save_crop(os.path.join(path,'cropped'+os.path.basename(path)))
    print("cropped all the images in "+os.path.basename(path))
def whileCapturing(img):
    print(json.loads(model(img)[0].tojson())[0]['box'])
    # result.save_crop(path, )
    # print(f'save to {path}')