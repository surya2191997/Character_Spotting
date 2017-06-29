# Character_Spotting

Spotting of Bali characters on manuscript images. The dataset contains about 10 manuscript images from which characters are segregated and annotated manually. This is train set for the classifier which contains about 19000 annotated character images. Final goal is : Given a querry image, spot the characters in the manuscript with the same annotation such that there is maximum overlap with the ground truth.

Images are preprocessed using bilateral filter and gamma correction . Gray scale image resized to 30 X 30 are used.

Approaches for Classification :

1. Simple Softmax Classifier
2. CNN (Yields highest accuracy of about 83 % for 70-30 split) 
3. Autoencoder for feature extraction, Softmax on top
