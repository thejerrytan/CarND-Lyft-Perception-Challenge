# IMG_SIZE = (320,800,3) # This is the original image size, - sky pixels and hood 
ORIGINAL_SIZE = (320, 800, 3)
IMG_SIZE = (160, 400, 3)
NUM_TO_SHOW = 10
LABEL_MAP = {
    0: None,
    1: "Buildings",
    2: "Fences",
    3: "Other",
    4: "Pedestrians",
    5: "Poles",
    6: "RoadLines",
    7: "Roads",
    8: "Sidewalks",
    9: "Vegetation",
    10: "Vehicles",
    11: "Walls",
    12: "TrafficSigns"
}
PREFIX = ""
DATA_X_FOLDER = "./*train/CameraRGB/"
DATA_Y_FOLDER = "./*train/CameraSeg/"
NUM_TO_SHOW = 10
ROAD_THRESHOLD = 0.90
VEHICLE_THRESHOLD = 0.10
START_Y = 200
END_Y = 520
BETA_ROAD = 0.5
BETA_VEHICLE = 2
UNET_ROAD_FILENAME = "unet_small_road_160_400.hdf5"
UNET_VEHICLE_FILENAME = "unet_large_vehicle.hdf5"