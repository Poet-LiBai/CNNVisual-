# coding: utf-8
 
from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation
from pylab import *
import keras
 
 
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
 
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
        title('feature_map_{}'.format(i))
 
    plt.savefig('feature_map.png')
    plt.show()
 
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")
 
 
def create_model():
    model = Sequential()
 
    # 第一层CNN
    # 第一个参数是卷积核的数量，第二三个参数是卷积核的大小
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
 
    # 第二层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
 
    # 第三层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    # 第四层CNN
    model.add(Convolution2D(9, 3, 3, input_shape=img.shape))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
 
    return model
 
 
if __name__ == "__main__":
    img = cv2.imread('001.jpg')
 
    model = create_model()
 
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
 
    visualize_feature_map(conv_img)

# coding: utf-8
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
 
 
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch):
    feature_map = img_batch
    print(feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
 
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
 
    plt.savefig('feature_map.png')
    plt.show()
 
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")
 
 
if __name__ == "__main__":
    base_model = VGG19(weights='imagenet', include_top=False)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
 
    img_path = '001.jpg'
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    block_pool_features = model.predict(x)
    print(block_pool_features.shape)
 
    feature = block_pool_features.reshape(block_pool_features.shape[1:])
 
    visualize_feature_map(feature)