## Classification of Traffic Signs in the Wild Using CNN

This project is to recognize the traffic signs in the wild (real-world) which is one of the main tasks for any self-driving car project. It is a computer vision classification problem that I’ve tackled by building a CNN model trained from scratch to do the job. The original dataset has more than **50,000 traffic sign images** with different sizes collected from the real-world, these images are categorized into 43 target labels. A cross validation technique using randomized grid search has been applied to find-tune the hyperparameters of the model. There are large variations in the visual appearance of the traffic signs in this dataset due to the weather conditions, illumination changes, rotations, the time of the day (night time or day time)…etc. and this was a challenge. My final solution model was able to achieve about **98.5%** prediction accuracy on the test dataset which has **12,630 instances**. I also converted my final solution model into a REST APIs to consume it later in a mobile or a web App.

## Project Summary

- The implementation of the CNN classifier is available in "Traffic_Signs_Classifier.ipynb" file.
 
- I converted the original dataset that has PPM files to a pickle files, one for training dataset and the other one for the test dataset. 

 - To check the code that I implemented to convert the original dataset to pickle files , please check the "PickleTrainingDataset.py" file which is already attached with the submission.

 - I have uploaded the two pickle files (train.pkl & test.pkl) to OneDrive and it could be reached through the following link: https://1drv.ms/f/s!Apt9CJrW-9NSghXkbEbbY50ZAe-H

- I saved the class labels of the traffic signs in npz file format called "traffic_sign_labels.npz"

- The Flask app of the solution model is in "TrafficSigns_WebApp" folder.

- The weights of the solution model is available in "saved_models" folder. This folder includes two main files as follows:
	- "weights.best.model.cv.hdf5" which has the weights before data augmentation.
	- "weights.best.model.optimized.hdf5" which has the weights after applying the data augmentation.


- The train and test datasets in a pickle format should be in "traffic-signs-dataset" folder as shown in the Jupyter notebook.


## Testing the final model against unseen traffic signs

<img src="https://github.com/osamadev/Traffic-Signs-CNN-Classifier/blob/master/Test_Results/test_results.png" >
