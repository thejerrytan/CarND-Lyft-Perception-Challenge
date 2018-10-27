import sys, json, base64, time
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import numpy as np
import cv2
import keras
from unet import Unet
from constants import *
import tensorflow as tf

file = sys.argv[-1]

if file == 'predict.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
        retval, buffer = cv2.imencode('.png', array)
        return base64.b64encode(buffer).decode("utf-8")


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