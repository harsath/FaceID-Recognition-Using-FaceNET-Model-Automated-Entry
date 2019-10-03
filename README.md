# FaceNet Model For Facial Recognition
## FaceNet is a state of the art Model for Facial Recognition. Despite significant recent advances in the field of face recognition, implementing face verification and recognition efficiently at scale presents serious challenges to current approaches. 

## How FaceNet Works?
FaceNet, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors. 
Our method uses a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches.

## How My Demo Works?
### :one: First: 
Dowload and Paste the FaceNet Pretrained Model Deep Learning Model And Save It Into `Model` Folder From (Here)[https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view]
### :two: Second: 
Take a Series Of Images Form a Camera and Store It in Train Images Folder And Then Run the Following Command. It Will Preprocess(MT-CNN) the data and Train the Model For Facial Recognition.
<img width="1171" alt="Screenshot 2019-06-18 at 12 40 11 PM" src="https://user-images.githubusercontent.com/30565388/59660652-be135180-91c6-11e9-8253-676c6c36f57e.png">

### :three: Third:
Once the Training of the Model is Done, Test the Model Via Live Inference From Webcame Or Real-Time Video. Type the Following Code
<img width="1171" alt="Screenshot 2019-06-18 at 12 39 05 PM" src="https://user-images.githubusercontent.com/30565388/59660825-177b8080-91c7-11e9-899d-1138656fcfbe.png">

## Features Of My FaceID:
During Live Inference Once A Face Is Detected On The Camera, It Automatically Fills A CSV File With Date and Person's Name. It Also Saves A Snap Of the Detected Face in Seperate Folder For Verification. This can Also be able to Send To a FireBase Database For Incorporating With an Android or iOS Applications.
## FaceNet Tensorflow Implementation credits to Dr.David Sandberg
https://github.com/davidsandberg
# Licence :clipboard:(Tensorflow implementation scipts)
Licensed Under <b>`GNU Affero General Public License v3.0`</b> Visit LICENCE file to know the Limitations.






