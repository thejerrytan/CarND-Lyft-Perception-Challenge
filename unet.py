import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Reshape, Lambda
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from constants import *
import tensorflow as tf
from keras import losses

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

class Unet(object):

    def __init__(self, train, test, beta, size, activation, model_filename, img_rows = 512, img_cols = 512, class_weight = None, num_epochs = 1):
        self.training_generator = train
        self.validation_generator = test
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.class_weight = class_weight
        self.num_epochs = num_epochs
        self.steps_per_epoch = None if train is None else len(train)
        self.model_filename = model_filename
        self.beta = beta
        self.size = size
        self.activation = activation

    def load_data(self):
        pass

    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols, 3))
        norm = Lambda(lambda x: x/255)(inputs)

        conv1 = Conv2D(self.size * 16, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(norm)
        conv1 = Conv2D(self.size * 16, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(self.size * 32, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(self.size * 32, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(self.size * 64, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(self.size * 64, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(self.size * 128, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(self.size * 128, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(self.size * 256, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(self.size * 256, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(self.size * 128, 2, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(self.size * 128, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(self.size * 128, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(conv6)
        
        up7 = Conv2D(self.size * 64, 2, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(self.size * 64, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(self.size * 64, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(self.size * 32, 2, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(self.size * 32, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(self.size * 32, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(self.size * 16, 2, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(self.size * 16, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(self.size * 16, 3, activation = self.activation, padding = 'same', kernel_initializer = 'he_normal')(conv9)
        
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        # conv10 = Reshape((conv10.get_shape()[1] * conv10.get_shape()[2], 1))(conv10)
        
        model = Model(inputs = inputs, outputs = conv10)

        # model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, sample_weight_mode = "temporal", metrics = [dice_coef])
        model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = [self.f2_score])
        # model.summary()
        
        return model


    # def train(self, force=True):
    #     model_checkpoint = ModelCheckpoint(self.model_filename, monitor='loss',verbose=1, save_best_only=True)
    #     try:
    #         self.model = load_model(self.model_filename, custom_objects={'dice_coef': dice_coef, 'loss': dice_coef_loss})
    #         if force:
    #             print('Training pre-existing model...')
    #             self.model.fit_generator(generator=self.training_generator,
    #                         validation_data=self.validation_generator,
    #                         use_multiprocessing=True,
    #                         workers=6, epochs=self.num_epochs, verbose=1, steps_per_epoch=self.steps_per_epoch,
    #                         callbacks=[model_checkpoint])
    #     except IOError as e:
    #         self.model = self.get_unet()
    #         print('Fitting model...')
    #         self.model.fit_generator(generator=self.training_generator,
    #                         validation_data=self.validation_generator,
    #                         use_multiprocessing=True,
    #                         workers=6, epochs=self.num_epochs, verbose=1, steps_per_epoch=self.steps_per_epoch,
    #                         callbacks=[model_checkpoint])

    def train(self, force=True):
        self.model = self.get_unet()
        model_checkpoint = ModelCheckpoint(self.model_filename, monitor='loss',verbose=1, save_best_only=True)
        if force:
            print('Training pre-existing model...')
            self.model.fit_generator(generator=self.training_generator,
                        validation_data=self.validation_generator,
                        use_multiprocessing=True,
                        workers=6, epochs=self.num_epochs, verbose=1, steps_per_epoch=self.steps_per_epoch,
                        callbacks=[model_checkpoint])
        else:
            try:
                self.model.load_weights(self.model_filename)
            except OSError as e:
                print('Fitting model...')
                self.model.fit_generator(generator=self.training_generator,
                                validation_data=self.validation_generator,
                                use_multiprocessing=True,
                                workers=6, epochs=self.num_epochs, verbose=1, steps_per_epoch=self.steps_per_epoch,
                                callbacks=[model_checkpoint])

    def precision(self, y_true_f, y_pred_f):
        tp = K.sum(y_true_f * y_pred_f)
        fp = K.sum(tf.cast(tf.logical_and(y_pred_f == 1, y_true_f == 0), tf.float32))
        return tp / (tp + fp) / 1.0

    def recall(self, y_true_f, y_pred_f):
        tp = K.sum(y_true_f * y_pred_f)
        fn = K.sum(tf.cast(tf.logical_and(y_pred_f == 0, y_true_f == 1), tf.float32))
        return tp / (tp + fn) / 1.0
        

    def f_score(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        p = self.precision(y_true_f, y_pred_f)
        r = self.recall(y_true, y_pred)
        return (1 + self.beta * self.beta) * (p * r) / (self.beta * self.beta * p + r)

    def f_loss(self, y_true, y_pred):
        return 1 - self.f_score(y_true, y_pred)

    def f2_score(self, y_true, y_pred):
        y_true = tf.cast(y_true, "int32")
        y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
        y_correct = y_true * y_pred
        sum_true = tf.reduce_sum(y_true, axis=1)
        sum_pred = tf.reduce_sum(y_pred, axis=1)
        sum_correct = tf.reduce_sum(y_correct, axis=1)
        precision = sum_correct / sum_pred
        recall = sum_correct / sum_true
        f_score = (1 + self.beta ** 2) * precision * recall / ((self.beta ** 2) * precision + recall)
        f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
        return tf.reduce_mean(f_score)
