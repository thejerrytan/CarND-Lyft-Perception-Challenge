import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pandas as pd
import keras
from unet import Unet
from constants import *

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
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    labels_new[hood_pixels] = 0
    # Return the preprocessed label image 
    return labels_new

# Convert to binary, still 3-channel image
def relabel(label_image, target_label):
    labels_new = np.zeros_like(label_image)
    target_pixels = (label_image[:,:,0] == target_label).nonzero()
    labels_new[target_pixels] = 1.0
    return labels_new

X_paths = glob.glob(PREFIX + DATA_X_FOLDER + "*.png")
y_paths = glob.glob(PREFIX + DATA_Y_FOLDER + "*.png")
print("%d training images , %d target imagess" % (len(X_paths), len(y_paths)))
print("Image size is (600, 800, 3)")
print("There are %d class labels before preprocessing." % (len(LABEL_MAP)))


### Augmentation functions

def augment_brightness(imageX, imageY):
    ### Augment brightness
    random_bright = .25+np.random.uniform()

    image1 = cv2.cvtColor(imageX,cv2.COLOR_RGB2HSV)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    
    image2 = cv2.cvtColor(imageY,cv2.COLOR_RGB2HSV)
    image2[:,:,2] = image2[:,:,2]*random_bright
    image2 = cv2.cvtColor(image2,cv2.COLOR_HSV2RGB)
    return image1, image2

def trans_image(imageX, imageY, trans_range):
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    
    rows,cols,channels = imageX.shape
    imageX_tr = cv2.warpAffine(imageX,Trans_M,(cols,rows))
    imageY_tr = cv2.warpAffine(imageY,Trans_M,(cols,rows))
    return imageX_tr, imageY_tr


def scale_image(imageX, imageY, scale_range):
    # Scale augmentation
    original_shape = imageX.shape
    halfy = original_shape[1] // 2
    halfx = original_shape[0] // 2
    scale_factor = np.random.uniform(1, scale_range)
    new_shape = (int(original_shape[1] * scale_factor), int(original_shape[0] * scale_factor))
    
    imgX = cv2.resize(imageX, new_shape)
    imgY = cv2.resize(imageY, new_shape)
    if scale_factor > 1:
        # BE CAREFUL! new_shape is (cols, rows) as required by opencv resize
        startx = (new_shape[1] // 2) - halfx
        endx = (new_shape[1] // 2) + halfx
        starty = (new_shape[0] // 2) - halfy
        endy = (new_shape[0] // 2) + halfy
        return imgX[startx:endx, starty:endy, :], imgY[startx:endx, starty:endy, :]
    else:
        return imgX, imgY

# rescale to bounding box of all isolated instances of the target label
# Does not preserve scale in x and y directions
def rescale_to_bb(imageX, imageY):
    original_shape = imageX.shape
    vehicle_pixels = np.where(imageY == 1)
    if len(vehicle_pixels[0]) > 0 and len(vehicle_pixels[1] > 0):
        xmin = min(vehicle_pixels[0])
        xmax = max(vehicle_pixels[0])
        ymin = min(vehicle_pixels[1])
        ymax = max(vehicle_pixels[1])
        resizedX = cv2.resize(imageX[xmin:xmax, ymin:ymax], (original_shape[1], original_shape[0]))
        resizedY = cv2.resize(imageY[xmin:xmax, ymin:ymax], (original_shape[1], original_shape[0]))
        return resizedX, resizedY
    else:
        return imageX, imageY

def flip_lr(imageX, imageY):
    flipped_imageX = cv2.flip(imageX, 1)
    flipped_imageY = cv2.flip(imageY, 1)
    return flipped_imageX, flipped_imageY

# rescale to bounding box given
# Does not preserve x:y aspect ratio
def rescale_to_bb_given(imageX, imageY, bb):
    original_shape = imageX.shape
    ul, lr = bb
    xmin, ymin = ul
    xmax, ymax = lr
    resizedX = cv2.resize(imageX[ymin:ymax, xmin:xmax], (original_shape[1], original_shape[0]))
    resizedY = cv2.resize(imageY[ymin:ymax, xmin:xmax], (original_shape[1], original_shape[0]))
    return resizedX, resizedY

MIN_BB_SIZE_X = 10
MIN_BB_SIZE_Y = 10
def get_bboxes(imageY):
    bboxes = []
    labels = label(imageY)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        min_x = np.min(nonzerox)
        min_y = np.min(nonzeroy)
        max_x = np.max(nonzerox)
        max_y = np.max(nonzeroy)
        if (max_x - min_x > MIN_BB_SIZE_X) and (max_y - min_y > MIN_BB_SIZE_Y):
            bboxes.append(((min_x, min_y), (max_x, max_y)))
    return bboxes

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        min_x = np.min(nonzerox)
        min_y = np.min(nonzeroy)
        max_x = np.max(nonzerox)
        max_y = np.max(nonzeroy)
        bbox = ((min_x, min_y), (max_x, max_y))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def is_vehicle_sideways(image, bb):
    if bb is None: return False
    else:
        ul, lr = bb
        xmin, ymin = ul
        xmax, ymax = lr
        # print((ymax - ymin) / (1.0 * (xmax - xmin)))
        if (ymax - ymin) / (1.0 * (xmax - xmin)) < 0.60: 
        # if height is lesser than length of car
            return True
        else:
            return False

def is_vehicle_headon(image, bb):
    if bb is None: return False
    else:
        grad = -image.shape[0]/image.shape[1]
        # As long as center of car lies above diagonal line from lower left to upper right of image
        # it is head-on car in opposite lane
        ul, lr = bb
        xmin, ymin = ul
        xmax, ymax = lr
        carX = (xmin + xmax) // 2
        carY = (ymin + ymax) // 2
        if carY < carX * grad + image.shape[0]:
            return True
        else:
            return False

# Build dataset
from keras.utils import Sequence

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X_paths, y_paths, target_label, batch_size=32, dim=(32,32), n_channels=1,
                 n_classes=10, shuffle=True, augmentation_factor=0):
        'Initialization'
        self.X_paths = X_paths
        self.y_paths = y_paths
        self.target_label = target_label
        self.dim = dim
        self.batch_size = batch_size
        self.dataset_size = len(self.X_paths)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation_factor = augmentation_factor
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.dataset_size * (self.augmentation_factor * 3 + 1) / self.batch_size))

    # Index must be the batch number from 0 to len(self)
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        effective_batch_size = self.batch_size // ((self.augmentation_factor * 3) + 1)
        indexes = self.indexes[index*effective_batch_size:(index+1)*effective_batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.dataset_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], 1), dtype=np.float32)
        
        # Generate data
        for i, idx in enumerate(indices):
            xpath = self.X_paths[idx]
            ypath = self.y_paths[idx]
            x_img = (mpimg.imread(xpath) * 255).astype(np.uint8)
            y_img = (mpimg.imread(ypath) * 255).astype(np.uint8)

            y_img = preprocess_labels(y_img)
            relabeled = relabel(y_img, self.target_label)

            # Crop out the sky and hood
            x_img = x_img[200:520, :, 0:3]
            relabeled = relabeled[200:520, :, 0:3]

            # Resize
            x_img = cv2.resize(x_img, (IMG_SIZE[1], IMG_SIZE[0]))
            relabeled = cv2.resize(relabeled, (IMG_SIZE[1], IMG_SIZE[0]))

            # Store sample
            baseIdx = i*(self.augmentation_factor*3+1) 
            X[baseIdx,] = x_img
            y[baseIdx,] = np.reshape(relabeled[:,:,0], (relabeled.shape[0], relabeled.shape[1], 1))
            
            # augmentation
            for j in range(0, self.augmentation_factor):
                x_brightness, y_brightness = augment_brightness(x_img, relabeled)
                X[baseIdx + (j * 3) + 1,] = x_brightness
                y[baseIdx + (j * 3) + 1,] = np.reshape(relabeled[:,:,0], (relabeled.shape[0], relabeled.shape[1], 1))
                x_translated, y_translated = trans_image(x_img, relabeled, 200)
                X[baseIdx + (j * 3) + 2] = x_translated
                y[baseIdx + (j * 3) + 2] = np.reshape(y_translated[:,:,0], (relabeled.shape[0], relabeled.shape[1], 1))
                x_flipped, y_flipped = flip_lr(x_img, relabeled)
                X[baseIdx + (j * 3) + 3] = x_flipped
                y[baseIdx + (j * 3) + 3] = np.reshape(y_flipped[:,:,0], (relabeled.shape[0], relabeled.shape[1], 1))

        return X, y

# Split into train and test datasets
if __name__ == "__main__":
    from itertools import compress
    FRACTION_TO_TRAIN = 0.7
    msk = np.random.rand(len(X_paths)) < FRACTION_TO_TRAIN

    X_paths_train = list(compress(X_paths, msk))
    X_paths_test = list(compress(X_paths, ~msk))
    y_paths_train = list(compress(y_paths, msk))
    y_paths_test = list(compress(y_paths, ~msk))

    print("Splitting into train and validation sets")
    print(len(X_paths_train))
    print(len(X_paths_test))
    assert(len(set(X_paths_train).intersection(set(X_paths_test))) == 0)
    assert(len(set(y_paths_train).intersection(set(y_paths_test))) == 0)

    # Parameters
    params = {'dim': (IMG_SIZE[0], IMG_SIZE[1]),
              'batch_size': 20,
              'n_classes': 2,
              'n_channels': 3,
              'shuffle': True,
              'augmentation_factor': 1
             }
    num_epochs = 20
    steps_per_epoch = int(len(X_paths_train)//params['batch_size'])

    training_generator_vehicle = DataGenerator(X_paths_train, y_paths_train, 10, **params)
    validation_generator_vehicle = DataGenerator(X_paths_test, y_paths_test, 10, **params)
    model_vehicle = Unet(training_generator_vehicle, validation_generator_vehicle, BETA_VEHICLE, 2, 'elu', UNET_VEHICLE_FILENAME, IMG_SIZE[0], IMG_SIZE[1], num_epochs=num_epochs)
    model_vehicle.train(force=True)

    training_generator_road = DataGenerator(X_paths_train, y_paths_train, 7, **params)
    validation_generator_road = DataGenerator(X_paths_test, y_paths_test, 7, **params)
    model_road = Unet(training_generator_road, validation_generator_road, BETA_ROAD, 1, 'relu', UNET_ROAD_FILENAME, IMG_SIZE[0], IMG_SIZE[1], num_epochs=num_epochs)
    model_road.train(force=True)
