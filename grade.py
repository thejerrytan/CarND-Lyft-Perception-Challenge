# Scoring function
import cv2
from unet import Unet
import sys
import time
import json
import base64
import glob
import numpy as np
import matplotlib.image as mpimg
from scipy import misc
from train import preprocess_labels, relabel, DataGenerator
from constants import *
import tensorflow as tf

#  returns binary arrays for grading
def process_frame_for_grading(frame, road_graph, vehicle_graph, road_threshold, vehicle_threshold):
    # predict
    cropped = frame[START_Y:END_Y,:]
    img_to_predict = np.reshape(cropped, (1, ORIGINAL_SIZE[0], ORIGINAL_SIZE[1], IMG_SIZE[2]))

    cropped = cv2.resize(cropped, (IMG_SIZE[1], IMG_SIZE[0]))
    img_to_predict_small = np.reshape(cropped, (1, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    
    t0 = time.time()
    x = road_graph.get_tensor_by_name("input_1:0")
    y_layer = road_graph.get_tensor_by_name("conv2d_23/Sigmoid:0")
    feed_dict = {x: img_to_predict_small}
    yhat_road = road_sess.run(y_layer, feed_dict = feed_dict)

    t1 = time.time()
    x = vehicle_graph.get_tensor_by_name("input_1:0")
    y_layer = vehicle_graph.get_tensor_by_name("conv2d_23/Sigmoid:0")
    feed_dict = {x: img_to_predict}
    yhat_vehicle = vehicle_sess.run(y_layer, feed_dict = feed_dict)
    t2 = time.time()

    print("Time taken to predict road: %.3fms" % ((t1-t0) * 1000.0))
    print("Time taken to predict vehicle: %.3fms" % ((t2 - t1) * 1000.0))
    
    # Upsize predictions
    yhat_road = cv2.resize(yhat_road[0,:,:,0], (ORIGINAL_SIZE[1], ORIGINAL_SIZE[0]))
    # yhat_vehicle = cv2.resize(yhat_vehicle[0,:,:,], (ORIGINAL_SIZE[1], ORIGINAL_SIZE[0]))

    # binarize
    binary_road_result = np.zeros((600, 800)).astype('uint8')
    binary_car_result = np.zeros((600, 800)).astype('uint8')
    binary_cropped_road = np.where(yhat_road > road_threshold, 1, 0).astype('uint8')
    binary_cropped_vehicle = np.where(yhat_vehicle > vehicle_threshold, 1, 0).astype('uint8')
    
    # overlap = np.where((binary_cropped_vehicle == 1) & (binary_cropped_road == 1))
    # binary_cropped_vehicle[overlap] = 1
    # binary_cropped_road[overlap] = 0

    binary_road_result[START_Y:END_Y, :] = binary_cropped_road
    binary_car_result[START_Y:END_Y, :] = binary_cropped_vehicle[0,:,:,0]
    return binary_car_result, binary_road_result

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
# vehicle_model = Unet(None, None, BETA_VEHICLE, 2, 'relu', UNET_VEHICLE_FILENAME, ORIGINAL_SIZE[0], ORIGINAL_SIZE[1], num_epochs=5)
# road_model = Unet(None, None, BETA_ROAD, 1, 'relu', UNET_ROAD_FILENAME, IMG_SIZE[0], IMG_SIZE[1], num_epochs=5)
# road_model.train(force=False)
# vehicle_model.train(force=False)

frames_processed = 0
EPISODE_NUM = "town02_episode_0012_train"
file_list_x = glob.glob("./%s/CameraRGB/*.png" % EPISODE_NUM)
file_list_y = glob.glob("./%s/CameraSeg/*.png" % EPISODE_NUM)
ans_data = {}
student_ans_data = {}
frame = 1
vehicle_sess = tf.Session(graph = vehicle_graph)
road_sess = tf.Session(graph = road_graph)

for xpath, ypath in zip(file_list_x[1:50], file_list_y[1:50]):
    rgb_frame = (mpimg.imread(xpath) * 255).astype(np.uint8)
    y_img = (mpimg.imread(ypath) * 255).astype(np.uint8)
    y_img = preprocess_labels(y_img)
    vehicle = relabel(y_img, 10)
    road = relabel(y_img, 7)
    binary_car_result, binary_road_result = process_frame_for_grading(rgb_frame, road_graph,
                                                                      vehicle_graph, ROAD_THRESHOLD, VEHICLE_THRESHOLD)
    student_ans_data[frame] = [binary_car_result, binary_road_result]
    ans_data[frame] = [vehicle[:,:,0], road[:,:,0]]
    frame += 1


Car_TP = 0 # True Positives
Car_FP = 0 # Flase Positives
Car_TN = 0 # True Negatives
Car_FN = 0 # True Negatives

Road_TP = 0 # True Positives
Road_FP = 0 # Flase Positives
Road_TN = 0 # True Negatives
Road_FN = 0 # True Negatives

Road_Car_Overlap = 0

for frame in range(1,len(ans_data.keys())+1):

    truth_data_car =  ans_data[frame][0]
    truth_data_road =  ans_data[frame][1]
    student_data_car = student_ans_data[frame][0]
    student_data_road = student_ans_data[frame][1]
    
    Road_Car_Overlap += np.sum(np.logical_and(student_data_car == 1, student_data_road == 1))

    Car_TP += np.sum(np.logical_and(student_data_car == 1, truth_data_car == 1))
    Car_FP += np.sum(np.logical_and(student_data_car == 1, truth_data_car == 0))
    Car_TN += np.sum(np.logical_and(student_data_car == 0, truth_data_car == 0))
    Car_FN += np.sum(np.logical_and(student_data_car == 0, truth_data_car == 1))

    Road_TP += np.sum(np.logical_and(student_data_road == 1, truth_data_road == 1))
    Road_FP += np.sum(np.logical_and(student_data_road == 1, truth_data_road == 0))
    Road_TN += np.sum(np.logical_and(student_data_road == 0, truth_data_road == 0))
    Road_FN += np.sum(np.logical_and(student_data_road == 0, truth_data_road == 1))

    frames_processed+=1


# Generate results
print("CAR:")
print("TP: %.7d, FP: %.7d, FN: %.7d" % (Car_TP, Car_FP, Car_FN))
print()
print("ROAD:")
print("TP: %.7d, FP: %.7d, FN: %.7d" % (Road_TP, Road_FP, Road_FN))
print()
print("Overlap: %.7d" % (Road_Car_Overlap))

Car_precision = Car_TP/(Car_TP+Car_FP)/1.0
Car_recall = Car_TP/(Car_TP+Car_FN)/1.0
Car_beta = 2
Car_F = (1+Car_beta**2) * ((Car_precision*Car_recall)/(Car_beta**2 * Car_precision + Car_recall))
Road_precision = Road_TP/(Road_TP+Road_FP)/1.0
Road_recall = Road_TP/(Road_TP+Road_FN)/1.0
Road_beta = 0.5
Road_F = (1+Road_beta**2) * ((Road_precision*Road_recall)/(Road_beta**2 * Road_precision + Road_recall))

print ("Car F score: %05.3f  | Car Precision: %05.3f  | Car Recall: %05.3f  |\n\
Road F score: %05.3f | Road Precision: %05.3f | Road Recall: %05.3f | \n\
Averaged F score: %05.3f" %(Car_F,Car_precision,Car_recall,Road_F,Road_precision,Road_recall,((Car_F+Road_F)/2.0)))

