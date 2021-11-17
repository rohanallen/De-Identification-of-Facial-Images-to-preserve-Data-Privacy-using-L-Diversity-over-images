# De-Identification-of-Facial-Images-to-preserve-Data-Privacy-using-L-Diversity-over-images
Averaging pixel values of images to de-identify and anonymize them to improve privacy.

Download all files and put it in a folder.

to map facial features 

sudo python3 detect_face_features_mapping.py --shape-predictor shape_predictor_68_face_landmarks.dat -i 3.jpeg

Averaging of facial features:only 2 images(3.jpg given twice)

sudo python3 detect_face_features_1.py --shape-predictor shape_predictor_68_face_landmarks.dat -i 1.jpeg -i2 3.jpeg -i3 3.jpeg

Averaging of facial features:3 imgs

sudo python3 detect_face_features_1.py --shape-predictor shape_predictor_68_face_landmarks.dat -i 1.jpeg -i2 2.jpeg -i3 3.jpeg

Description:

We map the facial features like eyes, nose, jawline and plot the coordinates using detect_face_features_mapping.py and then using detect_face_features_1.py
we do the same as above and average all the images together using their pixel co-ordinates to anonomize them and to de-identify them.
We will be making use of facial landmark detector implemented inside dlib produces 68 (x, y)-coordinates that map to specific facial structures. 
These 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset.
