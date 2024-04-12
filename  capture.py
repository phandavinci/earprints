import cv2
import os
import uuid
from earExtraction import forChangingWholeDir

className = input("Enter the className: ")
path = os.path.join('../data', className)
if not os.path.exists(path): os.makedirs(path)

cap  = cv2.VideoCapture(2)
while cap.isOpened():
    ret, frame = cap.read()  
    frame = frame[:360, :513, :]
    cv2.imshow('image', frame)   
    
    if cv2.waitKey(1) & 0XFF == ord('c'):
        img = os.path.join(path, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(img, cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
        print("Captured!!!")
    # if cv2.waitKey(1) & 0XFF == ord('p'):
    #     img = os.path.join(pospath, '{}.jpg'.format(uuid.uuid1()))
    #     cv2.imwrite(img, frame)
        
    if cv2.waitKey(0) & 0XFF == ord('q'):
        break
forChangingWholeDir(path)
cap.release()
cv2.destroyAllWindows()  