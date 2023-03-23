#coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np  
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from PIL import Image  
import cv2
import random
import os
from tqdm import tqdm  
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
kinit = 'glorot_normal'

matplotlib.use("Agg")

img_w = 384  
img_h = 384


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #print(img.shape)
    else:
        # img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
        # print(img.shape)
    return img


filepath ='/media/dy/Data_2T/CGP/Unet_Segnet/data/new-BUSI/1/Train_images/Augementa/'

def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + 'Train_images/'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'Train_images/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(filepath + 'Train_labels/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                # yield (train_data, train_label)
                yield (train_data, {"out1":train_label,"excitation7":train_label,"excitation6":train_label,"excitation5":train_label,"excitation4":train_label,"excitation3":train_label,"excitation2":train_label,"excitation1":train_label})
                train_data = []
                train_label = []
                batch = 0


# data for validation
def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'Train_images/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(filepath + 'Train_labels/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                # yield (valid_data, valid_label)
                yield (valid_data, {"out1":valid_label,"excitation7":valid_label,"excitation6":valid_label,"excitation5":valid_label,"excitation4":valid_label,"excitation3":valid_label,"excitation2":valid_label,"excitation1":valid_label})
                valid_data = []
                valid_label = []
                batch = 0


def squeeze_excitation_layer(data, skipdata, out_dim):
    '''
    SE module performs inter-channel weighting.
    '''
    concatenate = Concatenate()([data, skipdata])
    concatenate = Conv2D(out_dim, (3, 3), padding="same")(concatenate)
    concatenate = BatchNormalization()(concatenate)
    concatenate = LeakyReLU(alpha=0.01)(concatenate)

    squeeze = GlobalAveragePooling2D()(concatenate)
    excitation = Dense(units=out_dim // 4)(squeeze)
    # excitation = Dense(units=out_dim)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)

    scale = multiply([concatenate, excitation])

    data_scale = Concatenate()([data, scale])

    return data_scale

def sout(data,name):
    excitation0 = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(data)
    excitation0 = Activation('sigmoid')(excitation0)
    shape_g = K.int_shape(excitation0)
    excitation0 = UpSampling2D(size=(384// shape_g[1], 384 // shape_g[2]),name=name)(excitation0)
    return excitation0

def updata(filte, data, skipdata):
    shape_x = K.int_shape(skipdata)
    shape_g = K.int_shape(data)

    up1 = UpSampling2D(size=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2]))(data)

    Selective_data = squeeze_excitation_layer(up1, skipdata, filte)

    LeakyReLU2 = ConvBlock(Selective_data, filte)
    return LeakyReLU2

def ConvBlock(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data) #,dilation_rate=(4,4)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(filte, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    return LeakyReLU2

def DSEM(input_size):   
    inputs = Input(shape=input_size)

    Conv1 = ConvBlock(data=inputs, filte=64)

    pool1 = MaxPooling2D(pool_size=(2, 2))(Conv1)
    Conv2 = ConvBlock(data=pool1, filte=128)

    pool2 = MaxPooling2D(pool_size=(2, 2))(Conv2)
    Conv3 = ConvBlock(data=pool2, filte=128)

    pool3 = MaxPooling2D(pool_size=(2, 2))(Conv3)   
    Conv4 = ConvBlock(data=pool3, filte=256)

    pool4 = MaxPooling2D(pool_size=(2, 2))(Conv4)    
    Conv5 = ConvBlock(data=pool4, filte=256)

    pool5 = MaxPooling2D(pool_size=(2, 2))(Conv5)    
    Conv6 = ConvBlock(data=pool5, filte=512)

    pool6 = MaxPooling2D(pool_size=(2, 2))(Conv6)    
    Conv7 = ConvBlock(data=pool6, filte=512)

    pool7 = MaxPooling2D(pool_size=(2, 2))(Conv7)    
    Conv8 = ConvBlock(data=pool7, filte=1024)

    # 6
    up1 = updata(filte=512, data=Conv8, skipdata=Conv7)
    excitation1 = sout(data=up1,name='excitation1')
    # 12
    up2 = updata(filte=512, data=up1, skipdata=Conv6)
    excitation2 = sout(data=up2,name='excitation2')
    # 25
    up3 = updata(filte=256, data=up2, skipdata=Conv5)
    excitation3 = sout(data=up3,name='excitation3')
    # 48
    up4 = updata(filte=256, data=up3, skipdata=Conv4)
    excitation4 = sout(data=up4,name='excitation4')
    # 96
    up5 = updata(filte=128, data=up4, skipdata=Conv3)
    excitation5 = sout(data=up5,name='excitation5')
    # 192
    up6 = updata(filte=128, data=up5, skipdata=Conv2)
    excitation6 = sout(data=up6,name='excitation6')
    # 384
    up7 = updata(filte=64, data=up6, skipdata=Conv1)
    excitation7 = sout(data=up7,name='excitation7')

    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(up7)
    out1 = Activation('sigmoid',name='out1')(outconv)

    model = Model(inputs=inputs, outputs=[out1,excitation7,excitation6,excitation5,excitation4,excitation3,excitation2,excitation1])
    return model

def BCE():
    def dice(y_true, y_pred):
        return tf.keras.metrics.binary_crossentropy(y_true, y_pred)
    return dice

def train(args): 
    EPOCHS = 50
    BS = 12


    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        model = DSEM(input_size=(384, 384, 3))
        model.summary()

    model.compile(loss={'out1': BCE(), 'excitation7': BCE(), 'excitation6': BCE(), 'excitation5': BCE(), 'excitation4': BCE(), 'excitation3': BCE(), 'excitation2': BCE(), 'excitation1': BCE()},
    loss_weights={'out1': 1.0, 'excitation7': 1.0, 'excitation6': 1.0, 'excitation5': 1.0, 'excitation4': 1.0, 'excitation3': 1.0, 'excitation2': 1.0, 'excitation1': 1.0}, metrics=["accuracy"], optimizer=Adam(lr=1e-3))

    checkpointer = ModelCheckpoint(os.path.join('./model/BUSI/11/', 'model_{epoch:03d}.hdf5'), monitor='val_acc', save_best_only=False, mode='max')

    tensorboard = TensorBoard(log_dir='./logs/BUSI/11/', histogram_freq=0, write_graph=True, write_images=True)

    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=[checkpointer, tensorboard])   #,max_q_size=1
  
def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-m", "--save_dir", default="/media/dy/Data_2T/CGP/Unet_Segnet/method/SKNet/model/DatasetB/4/",
    #                 help="path to output model")
    args = vars(ap.parse_args()) 
    return args


if __name__=='__main__':  
    args = args_parse()
    train(args)  
    #predict()  
