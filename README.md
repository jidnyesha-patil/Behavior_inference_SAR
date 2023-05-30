# Behavior_inference_SAR
## Inference behavior for a socially assistive robot

Face and pose recognition can help determine behavioral aspects of patients during ABA therapy.The main objective of this project is to track the head and limbs of the person of interest in the video and live feed to infer the behavior portrayed. This data can be used to identify the person of interest (PoI) and detect the pose during the entire therapy session. The pose data can be used to infer whether the PoI was focused on the task at hand or distracted. As of now, only two inferences have been derived but this can be extended later on to include more poses. Also, the data collected while detecting the pose can be used to track the limbs and head of the PoI.

### Generate Database
Every new person who is admitted into ABA therapy will get their face profile recorded by taking a series of pictures of them with different face orientations. These pictures are then labelled with the person’s name and stored in the database for creation of a face recognition model. To get a dataset for pose recognition, the subjects were asked to mimic writing while sitting on a chair in front of the camera. This pose was considered to be a basis for recognition. Actual ABA therapy sessions can be used for better performance. Five minute recordings are used for further analysis

### Face Recognition
To recognize the faces, two different approaches were tested in this project. The first, using a pre-trained FaceNET model, and the second using Principal Component Analysis. 
#### Approach 1: FaceNET
FaceNET model is an open-source deep learning model available on GitHub. It takes 60x60 pixel images as input and generates 1x168 feature embeddings. These feature embeddings are generated for every person of interest using the database of their face profile created in the introduction phase. Then these embeddings are used to train a Support Vector Machine Classifier(SVM classifier). Steps taken for this approach:
1. Pre-process image dataset: image frame is fed into an MTCNN face detector module to get cropped out face images of the person of interest.
2. Convert these face images in a 60x60 pixel array with a normalized value.
3. Apply them to the FaceNET model and generate feature embeddings.
4. Use these feature embeddings to train a SVM classifier.
5. Use this SVM model to predict the right person of interest.

#### Approach 2: Principal Component Analysis
Principal Component Analysis is chosen since it is a preferred method for reducing computation. It uses the image data projected on a small feature space thus providing dimensional reduction.
Steps taken for PCA approach:
1. Pre-process image dataset : Crop around facenusing Haar Cascade and convert to grayscale
2. Store image dataset as csv file : Using PIL Library getdata() function 
3. Apply PCA without n_components
4. Extract eigenfaces from PCA
5. Perform PCA

PCA without n_components
Train dataset using scikit learn and deduce a suitable value for 'n_components'. 
