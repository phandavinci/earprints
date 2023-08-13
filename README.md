# Ear-Biometrics
<img src="readmefiles/Creative Studio Name LinkedIn Article Cover Image.png">

Have you ever thought about the uniqueness of your ears? 

No two ears are exactly alike, even among identical twins. This is what makes ear biometrics such an intriguing field of study. 
## What is Ear-Biometrics?
Ear biometrics refers to the usage of ear shape and features as a means of identification or authentication. Just like fingerprints, ears can be used to accurately identify someone.
## How ear biometrics differ from other biometrics?
Covid-19 has an inevitable impact on biometric systems such as wearing masks is a considerable impact on facial recognition.

To overcome the impact of the pandemic, EAR is the best biometrics alternative because ear biometrics are non-intrusive. Unlike fingerprints, it does not require physical contact with the device.
## Packages 
![tensorflow](https://img.shields.io/badge/TensorFlow-grey?style=for-the-badge&logo=tensorflow)

![Matplotlib](https://img.shields.io/badge/matplotlib-grey?style=for-the-badge&logo=matplotlib)

![python](https://img.shields.io/badge/Python-grey?style=for-the-badge&logo=python)

![opencv](https://img.shields.io/badge/opencv-grey?style=for-the-badge&logo=opencv)

![Keras](https://img.shields.io/badge/Keras-grey?style=for-the-badge&logo=keras)
## Siamese Network

Siamese Neural Network is a neural network architecture that consists of two identical sub-networks that share weights. It is commonly used for comparing or matching two inputs.

<img src="readmefiles/siamese .png">

## Training Set with Tripet Loss

<img src="readmefiles/triplet loss.png">

## Datasets

### IITD II Dataset(Negative Images)
The IIT Delhi ear image database consists of the ear image database collected from the students and staff at IIT Delhi, New Delhi, India.The currently available database is acquired from the 121 different subjects and each subject has at least three ear images.Recently, a larger version of ear database (automatically cropped and normalized) from 212 users with 754 ear images is also integrated and made available on request.

### Database Setup(Positive and Anchor Images)
#### Create Folder Structures
##### Setup Paths
```bash
    POS_PATH = os.path.join('data', 'positive')
    NEG_PATH = os.path.join('data', 'negative')
    ANC_PATH = os.path.join('data', 'anchor')
```
##### Make Directory
```bash
    os.makedirs(POS_PATH)
    os.makedirs(NEG_PATH)
    os.makedirs(ANC_PATH)
```
#### Camera Access (Positive and Anchor Images)
```bash
        cap  = cv2.VideoCapture(0)
     while cap.isOpened():
        ret, frame = cap.read()  
        frame = frame[:360, :513, :]
        cv2.imshow('image', frame)   
        
       if cv2.waitKey(1) & 0XFF == ord('a'):
            img = os.path.join(anchor, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(img, frame)
            
        if cv2.waitKey(1) & 0XFF == ord('p'):
            img = os.path.join(pospath, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(img, frame)
            
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()    
```
## Research Paper Work
For a detailed technical explanation of the ear biometrics algorithms used in this project, you can refer to our [PDF Documentation](readmefiles\Earbiometrics Research Paper.pdf)

## Contribution

I welcome contributions to improve the project! To contribute:

1. Fork the repository.
2. Create a new branch for your feature/fix: `git checkout -b feature-name`
3. Make your changes and commit them: `git commit -m "Description of changes"`
4. Push to your forked repository: `git push origin feature-name`
5. Create a pull request, describing your changes and their purpose.




