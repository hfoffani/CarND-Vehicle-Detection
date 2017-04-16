# Vehicle Detection Project

The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled set of images.
* Also apply a color transform and binned color features, as well as histograms of color and append all of them.
* Normalize and randomize training and testing sets.
* Train a Linear SVM classifier.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image10]: ./output_images/car_notcar.png
[image03]: ./output_images/sliding-windows.png
[image40]: ./output_images/pipeline0.png
[image41]: ./output_images/pipeline1.png
[image42]: ./output_images/pipeline2.png
[image43]: ./output_images/pipeline3.png
[image50]: ./output_images/search_with_heat0.png
[image51]: ./output_images/search_with_heat1.png
[image52]: ./output_images/search_with_heat2.png
[image53]: ./output_images/search_with_heat3.png
[image54]: ./output_images/search_with_heat4.png
[image55]: ./output_images/search_with_heat5.png
[image07]: ./output_images/pipeline5.png
[video1]: ./vehicle_detection.mp4

### Histogram of Oriented Gradients (HOG)


I started by reading in all the `vehicle` and `non-vehicle` images. The source code can be found in the cell under the **Load the data sets** title in the attached IPython notebook `vehicle-detection.ipynb`. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car or NOT][image10]

I then explored different color spaces and different HOG parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). The source code is in the `single_img_features` and `get_hog_features` functions in the cell titled **Helper functions**.

The one I am using is `YCrCb` color space and HOG parameters of `orientations == 11`, `pixels_per_cell == 8`, `cells_per_block == 2` in all three channels.


I have tried several combinations of HOG parameters. First I used RGB color space `orientations == 9` and while the accuracy was not bad it proved to be very slow. But I consider these parameters as my baseline for every other setup. I tested LUV color space but it generated negative numbers in some channel (probably a bug?) which makes `hog` generate NaN (not-a-number) and later breaks the classifier.

I have also used Jupyter's line profiler a lot (there are reference in the notebook source code.) It showed me that most of the processing time was spent in `skimage.hog()`. That led me to change from skimage's hog to the one implemented in cv2.

The final feature set consists of HOG over all three channels of `YCrCb` color space with `orientations == 11`, `pixels_per_cell == 8`, `cells_per_block == 2` plus the spatial features. Going from 9 to 11 orientations also allowed me to *avoid color histograms features* while keeping the same accuracy 99.1%

This fecture vector has a length of 9,540.

I have chosen the SVM algorithm because it is very fast at the classification step. Its training performance is quite good too provided the number of observations is not big. My 2008 PC handled a few thousands very well training 15,000 observations of 10,000 features in 40 seconds. It also shines when the labels in the dataset are balanced as is our case.

The cells under the **Train and test Model** title contains the code for training the model, saving and loading for future use and testing the accuracy. It results in:

```
Test Accuracy of SVC = 99.10%
False positives 0.23%
False negatives 0.68%
```

Features were scaled to zero mean. See **Normalization and randomization** cell. The test set is 20% of the labeled dataset.

Using cv2.HOG kept me having *the same* feature extraction function for training the model and classifying the video frames. Subtle bugs hard to fix are prone to appear when slightly different functions are used when training and run time.

### Sliding Window Search

I used 6 sampled images from a video with different light and objects conditions as a test bed. My priority was to keep false positive to a minimum while at the same time having boxes hitting the nearest cars. False positives do not have to be perfect though, a following step would take care of them.

I restricted the search of the sliding windows within a strip on the bottom half of the video frame. We do not expect to find cars flying above the horizon or resting over the hood (hopefully!).

I tried several combinations of window scales and overlap. I started with a 96 pixels square window with 50% overlap and gradually added more scales (96, 128 and 256) and overlaps (70, 75). I finally settled with five scales (80, 96, 112, 128 and 160) and a 60% overlap.

The functions `slide_window` and `search_windows` which implement this part can be found in the cell under the **Search windows** title.

![Sliding Windows][image03]


The final test images show a good result detecting cars and no false positives. A trick I used to improve the classifier performance was to cache the creation of the `cv2.HOGDescriptor` function.

![Pipeline][image40]
![Pipeline][image41]
![Pipeline][image42]
![Pipeline][image43]

----

### Video Implementation


[![Click to view!](https://img.youtube.com/vi/FUPRVKCp3iI/0.jpg)](https://www.youtube.com/watch?v=FUPRVKCp3iI)


For each sliding window that the classifier successfully detects a car I overlapped each box and created a heatmap. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap assuming that each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

I applied a similar technique during the video rendering where a threshold was applied: a zone is marked as containing a car if it was detected during 15 consecutive frames.

**Here are six frames and their corresponding heatmaps with label heatmap boxes:**

![With heatmap][image50]
![With heatmap][image51]
![With heatmap][image52]
![With heatmap][image53]
![With heatmap][image54]
![With heatmap][image55]


**Here the resulting bounding boxes are drawn onto the last frame in the series:**

![Bounding boxes][image07]


----

### Discussion

The biggest problem I had was the slow rendering time. It makes tunning the algorithm a tedious and time consuming process.

Obviously I would expect that my implmentation will suffer in night conditions. It will also fail to detect motorcycles and trucks as it has not been trained with these. Besides, in urban roads, other objects are expected so a lot of false positives and false negatives will appear.

There are some easy steps to take to improve the speed of the rendering. Using only cv2 functions during feature extraction is one of them. A lot of time is wasted converting matrices from a 0-255 scale to a 0-1 and back. Currently there are 5 window scales, reducing to one or two will directly cut by half or by fifth the processing time.

Also more training data is needed. Samples of motorcycles, trucks, light conditions will help a lot.

