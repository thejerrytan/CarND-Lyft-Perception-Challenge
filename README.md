## Lyft Perception Challenge

---
This project submission is for the [image segmentation challenge](https://www.udacity.com/lyft-challenge) organized by Lyft and Udacity hosted in May 2018.


**Lyft Perception Challenge**

The goals / steps of this project are the following:

* Given a 600 x 800 x 3 RGB image, output a pixel-wise segmantation map of road and vehicles
* Achieve real-time inference capability by keeping FPS > 10
* The precision and recall score for road and vehicle mask is evaluated separately to arrive at a beta weighted f2 score, where F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)Score is measured with beta = 2 for vehicle and beta = 0.5 for road. Final score is average of f2 score for road and vehicle - Penalty, where penalty = (10 - FPS) > 0 is penalty for running at a frame rate lower than 10

[//]: # (Image References)

[unet_architecture]: ./report/u-net-architecture.png "Unet architecture"
[cropped_leaderboard]: ./report/cropped_leaderboard.png "Top 10 leaderboard"
[training_dataset_visualization]: ./report/training_dataset_visualization.png "randomly selected images from training dataset"
[augmentation_visualization]: ./report/augmentation_visualization.png "Output of augmentation"
[datagenerator_output]: ./report/datagenerator_output.png "Output of Data-generator"
[unet_output_visualization]: ./report/unet_output_visualization.png "Ground Truth v.s. Image masks generated from Unet"

---

### File structure
- Example/ : directory containing example grading scripts provided by the organizers
- report/ : directory containing images for this README
- README.md : this file
- constants.py : python file containing constants
- environment-2.yaml : conda environment file for running keras_to_tensorflow script (see below)
- environment.yaml : conda environment file for this repo
- grade.py: the original scoring file takes output of predict.py, decodes it and matches against the ground truth to arrive at a F2 score. I have modified this to directly process a video file and grade it at the end.
- predict.py: run this script with path to video file as 2nd argument and pipe the output to results.json. The output can be used to by the organizer's scoring script to come up with a score for submission.
- train.py : run this script to train Unet against data in train/ folder
- unet.py : contains model definitions for unet architecture, as well as datagenerators
- unet_large_vehicle.hdf5 : unet large (base depth 32) for vehicle, weights only
- unet_large_vehicle.hdf5.pb : unet large vehicle frozen TF graph-def file
- unet_small_road.hdf5 : unet small (base depth 16) for road, weights only
- unet_small_road_160_400.hdf5 : unet small road for 160 x 400 images (downsized by factor of 2)
- unet_small_road_160_400.hdf5.pb : unet small road for 160 x 400 images, frozen TF graph-def file
- unet_vehicle.hdf5 : earliest trained unet with base depth 16 and full-sized image
- vehicle-detection.ipynb: Jupyter notebook for running the whole workflow end-to-end, interactively. The python files above are needed because the provided workspace with 50 free GPU hours is a bare metal linux server with no GUI.
- test_video.mp4 : the test_video that was provided for training purposes
- test_video_out.mp4 : output of my neural network on test_video.mp4
- actual_test_video.mp4 : the actual test video that was run for the final leaderboard scores
- actual_video_out.mp4 : the output of my neural network on actual test video
- output_video/ : directory containing segmented videos of some of my training data

### Results
The final leaderboard standings as of the official end of the competition on 3rd June 2018 is as folows:

![alt text][cropped_leaderboard]

Unfortunately, I didn't save the leaderboard scores for myself before they closed off access to the workspace and I have to rely on my memory for this.

I submitted my results under the pseudoname bankai, which is ranked 36 (roughly) with a total score of ~89.2 and FPS of 6.9 (thus suffering a penalty of 3.1).

Many of the top 10 contenders have since uploaded their code and written posts on how they achieved their results. I have learnt tremendously from them and from participating in this challenge. I look forward to more of such challenges in the future! :)

Check them out here:
- https://www.linkedin.com/pulse/how-i-won-lyft-perception-challenge-asad-zia/ (winner)
- https://medium.com/@jsaustin/lyft-perception-challenge-2nd-place-finish-8fcacb86f9fa (2nd-place)
- https://github.com/mirouxbt/Lyft-Perception-Challenge
- https://github.com/bearpelican/lyft-perception-challenge
- https://github.com/sagarbhokre/LyftChallengeV2

### Dataset collection

The original training dataset given to us consist of 1000 RGB 800 x 600 images and 1000 3 channel images with labels encoded in the R channel. There are 13 classes and as advised by Udacity, the host of the competition, we should preprocess the labels in order to remove the labels of the car hood that occupies the bottom 280 pixels of the image using this code:

```
def preprocess_labels(label_image):
    labels_new = np.copy(label_image)
    # Identify lane marking pixels (label is 6)
    lane_marking_pixels = (label_image[:,:,0] == 6).nonzero()
    # print(lane_marking_pixels)
    # Set lane marking pixels to road (label is 7)
    labels_new[lane_marking_pixels] = 7

    # Identify all vehicle pixels
    vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
    # Isolate vehicle pixels associated with the hood (y-position > 496)
    hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices],
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    labels_new[hood_pixels] = 0
    # Return the preprocessed label image 
    return labels_new
```
#### 1. Collecting more data

I installed the Carla simulator, ran the python client for 13 episodes, each time with a different weather setting, and repeated this with town02 map. Each episode yields 300 images, for a total of 13 * 2 * 300 more images. Furthermore, thanks to @chinkiat on #Lyft-Challenge slack channel, for sharing his dataset of 1000 more images with cars and pedestrians, which I included in the training set.

Here is a visualization of what the dataset looks like:

![alt text][training_dataset_visualization]

I have not uploaded the training dataset as the zip folder itself is over 5GB. PM me or raise a pull request if you want to get your hands on the dataset if you are trying to reproduce my results.

#### 2. Hard Negative Mining

Examining the output of trained networks revealed that it tended to classify pedestrians as cars. Red umbrellas carried by pedestrains in rainy weather was also wrongly classified as vehicles. Thus, began the tedious process of going through every segmantic segmantation video for 12 episodes, identifying frames where the network performed badly and adding those frames back to the dataset.

### Augmentation

I tried 3 basic augmentations - adjusting brightness, random translation in the x and y directions and flipping the image about the vertical axis. The last augmentation would help in situations where the vehicle is turning from left to right and the neural network would be able to learn the symmetric scenario of turning right to left.

I tried other techniques like taking the bounding boxes of head-on vehicles on the opposite lane, as well as turning vehicles, and blowing them up to the full image size, the rationale being 1) those are rare scenarios in an already imbalanced dataset of way more pixels classified as road rather than vehicles, 2) the size of vehicles is inversely related to number of vehicles - for every image, you can squeeze in more smaller vehicles further away from the camera while you can only fit a few large vehicles the closer they appear to the camera. This was my attempt at balancing the distribution of classes in the dataset, however, the network performed worst, thus I did not continue with this plan.

Below is the result of augmentation.

![alt text][augmentation_visualization]

And here is what the output of datagenerator looks like, just for sanity checking sake.

![alt text][datagenerator_output]

### Architecture

The architecture I chose is a very popular and effective encoder-decoder architecture - Unet, which has proven to achieve state-of-the-art results with semantic segmentation tasks.

![alt text][unet_architecture]

In order to optimize for both speed (FPS) and accuracy (F2 scores), there were a few hyperparameters that I can tune - the base depth (number of channels in the first and last convolution layers), as well as the size of the input image. The original Unet had base depth of 64 channels, which leads to a whooping 4 million parameters to train, which would kill real-time performance as well as training times. Turns out that a larger Unet leads to higher accuracy but lower FPS. A larger input image size also leads to higher accuracy but lower FPS. Since the f_score for road is always signicantly higher, it was possible to sacrifice accuracy for the road in exchange for lower FPS and more complicated unet models to improve accuracy scores for vehicles.

In the end, I settled for base depth of 16, input image size of (160 x 400 x 3) for the road unet and base depth of 32, input image size of (320, 800, 3) for the vehicle unet.

ELU - exponential linear units was used for the vehicle unet and found to achieve faster convergence.

### Approach

I trained 2 separate unets, one dedicated to segmenting road while the other is dedicated to segmenting vehicles. The problem becomes a binary classification problem and I was able to use sigmoid as activation for last layer and dice coefficient loss = 1 - dice coefficient, as my loss function, which has been said to achieve better results than binary_crossentropy loss.

Since the value of each output represents a probability of the pixel belonging to the positive class, I have another hyperparameter - the threshold above which to consider the pixel as positive class, to tune. A value of 0.9 for road which a value of 0.10 for vehicle achieves the highest f2 score.

Another advantage of training 2 unets is it gives me the freedom to have 2 independent architectures for each of the network, which allows me to make careful tradeoffs between model complexity and accuracy.

I ran through all images in the training set and realized there will never be any vehicle below pixel 200, hence, I can crop away the sky and the hood ```image[200:520, :,:]``` and reduce the size of the input image to (320 x 800 x 3)

Normalizing the image to range from 0 - 1 improves training as the range of values for the input matches the range for the output. Activation functions are well behaved for the range of values that we have chosen as well.

### Optimizing FPS

1. Use a different encoding function
```python
def encode(array):
        retval, buffer = cv2.imencode('.png', array)
        return base64.b64encode(buffer).decode("utf-8")
```

2. Use opencv's videocapture

```python
cap = cv2.VideoCapture(file)

while True:
    ret, bgr_frame = cap.read()
    if not ret: break
    
    cropped = bgr_frame[START_Y:END_Y,:,:]
    
    rgb_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    binary_car_result, binary_road_result = process_frame(rgb_frame, road_graph, vehicle_graph, ROAD_THRESHOLD, VEHICLE_THRESHOLD)

    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    
    # Increment frame
    frame+=1
```

3. Freeze tensoflow model and use it later for inference

Thanks to [amir-abdi](https://github.com/amir-abdi/keras_to_tensorflow), who provided a script to convert h5 model files to tensorflow protobuf GraphDef files which has been optimized for inference. I was able to load it in predict.py later to perform inference at a much faster speed.

```python
# Load model
frozen_vehicle_graph = "./unet_large_vehicle.hdf5.pb"
frozen_road_graph = "./unet_small_road_160_400.hdf5.pb"

with tf.gfile.GFile(frozen_vehicle_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as vehicle_graph:
    tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )
with tf.gfile.GFile(frozen_road_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as road_graph:
    tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )

vehicle_sess = tf.Session(graph = vehicle_graph)
road_sess = tf.Session(graph = road_graph)

def process_frame(cropped, road_graph, vehicle_graph, road_threshold, vehicle_threshold):
    # predict
    img_to_predict = np.reshape(cropped, (1, ORIGINAL_SIZE[0], ORIGINAL_SIZE[1], IMG_SIZE[2]))

    cropped = cv2.resize(cropped, (IMG_SIZE[1], IMG_SIZE[0]))
    img_to_predict_small = np.reshape(cropped, (1, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))

    x = road_graph.get_tensor_by_name("input_1:0")
    y_layer = road_graph.get_tensor_by_name("conv2d_23/Sigmoid:0")
    feed_dict = {x: img_to_predict_small}
    yhat_road = road_sess.run(y_layer, feed_dict = feed_dict)

    x = vehicle_graph.get_tensor_by_name("input_1:0")
    y_layer = vehicle_graph.get_tensor_by_name("conv2d_23/Sigmoid:0")
    feed_dict = {x: img_to_predict}
    yhat_vehicle = vehicle_sess.run(y_layer, feed_dict = feed_dict)

    # Upsize predictions
    yhat_road = cv2.resize(yhat_road[0,:,:,0], (ORIGINAL_SIZE[1], ORIGINAL_SIZE[0]))
    # yhat_vehicle = cv2.resize(yhat_vehicle[0,:,:,], (ORIGINAL_SIZE[1], ORIGINAL_SIZE[0]))

    # binarize
    binary_road_result = np.zeros((600, 800)).astype('uint8')
    binary_car_result = np.zeros((600, 800)).astype('uint8')
    binary_cropped_road = np.where(yhat_road > road_threshold, 1, 0).astype('uint8')
    binary_cropped_vehicle = np.where(yhat_vehicle > vehicle_threshold, 1, 0).astype('uint8')

    # In case of overlap, we predict vehicle because its much more likely that a vehicle 
    # is occluding the road
    # overlap = np.where((binary_cropped_vehicle == 1) & (binary_cropped_road == 1))
    # binary_cropped_vehicle[overlap] = 1
    # binary_cropped_road[overlap] = 0

    binary_road_result[START_Y:END_Y, :] = binary_cropped_road
    binary_car_result[START_Y:END_Y, :] = binary_cropped_vehicle[0,:,:,0]

    return binary_car_result, binary_road_result

answer_key = {}

# Frame numbering starts at 1
frame = 1
cap = cv2.VideoCapture(file)

while True:
    ret, bgr_frame = cap.read()
    if not ret: break
    
    cropped = bgr_frame[START_Y:END_Y,:,:]
    
    rgb_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    binary_car_result, binary_road_result = process_frame(rgb_frame, road_graph, vehicle_graph, ROAD_THRESHOLD, VEHICLE_THRESHOLD)

    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    
    # Increment frame
    frame+=1

# Print output in proper json format
print (json.dumps(answer_key))
```


---

### Video Implementation

Watch my videos at output_videos directory and test_video_out.mp4 and actual_video_out.mp4. I will be uploading them to Youtube also.

For the competition, I used unet_small_road_160_400.hdf5 and unet_large_vehicle.hdf5, which resulted in less accurate road mask but faster FPS.

For the video, I used the best models I have - unet_small_road.hdf5 and unet_large_vehicle.hdf5 in order to generate the videos. It takes about 5s per frame on my 2012 CPU only MacBook Air.

This is the code for generating video out of the input video files:

```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML

project_out_file = 'actual_video_out.mp4'
clip_test = VideoFileClip('actual_test_video.mp4')
clip_test_out = clip_test.fl_image(process_frame)
%time clip_test_out.write_videofile(project_out_file, audio=False)
```

And this is the code to embed a HTML5 player in the ipython notebook and view it:

```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_out_file))
```

---

### Discussion

On hindsight, I could have invested more effort in optimizing FPS. I was getting a penalty of 3 points, which would put me at total score of ~91-92, comfortably within the top 10 range. In fact, looking at the approaches of the top 25 contenders, we were all using the right techniques, thus the only difference separating the top 10 were hacks / optimizations specifically used to exploit this particular dataset. The multi-processing optimization techniques by the winner Asad were clearly masterful and separated him from the rest of the pack.