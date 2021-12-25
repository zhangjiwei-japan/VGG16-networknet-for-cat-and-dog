import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 指定GPU编号 Specify GPU number
#Read the data set, and modify the data set path
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="D:/DATASETS/cats_and_dogs_filtered/train",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="D:/DATASETS/cats_and_dogs_filtered/validation", target_size=(224,224))

def vgg16(): 
    # 使用序贯式模型 Use sequential model
    model = Sequential()
    # 两个3*3*64卷积核 + 一个最大池化层 Two 3*3*64 convolution kernels + a maximum pooling layer
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # 两个3*3*128卷积核 + 一个最大池化层 Two 3*3*128 convolution kernels + a maximum pooling layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # 三个3*3*56卷积核 + 一个最大池化层
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # 三个3*3*512卷积核 + 一个最大池化层
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # 三个3*3*512卷积核 + 一个最大池化层
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    # The Flatten layer is used to "flatten" the input, that is, to make the multi-dimensional input one-dimensional, and it is often used in the transition from the convolutional layer to the fully connected layer. Flatten does not affect the size of the batch.
    # 连接三个全连接层Dense，最后一层用于预测分类。Predictive classification
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    return model

import matplotlib.pyplot as plt   
def show_plot(hist):
    plt.plot(hist.history["acc"])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
    plt.show()

# 定义模型和精度计算方式 Define the model and accuracy calculation method
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
def train_model():
    log_dir = "logs/"
    filepath="model_{epoch:02d}-{val_accuracy:.2f}.h5"
    checkpoint = ModelCheckpoint(log_dir +filepath, monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
    model = vgg16()
    # 打印模型结构 Print model structure
    model.summary() 
    # 定义模型优化器， 使用分类交叉熵损失 Define model optimizer, use classification cross entropy loss
    from keras.optimizers import Adam
    opt = Adam(lr=0.000001)
    # model = multi_gpu_model(model, 2)  #GPU个数为2 The number of GPUs is 2
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print("start training")
    ts = time.time()
    # 训练模型并计算精度 Train the model and calculate the accuracy
    hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata,\
                            validation_steps=10,epochs=100,callbacks=[checkpoint, early])
    print("start training",time.time()-ts)
    # show_plot(hist)

from keras.preprocessing import image
from keras.models import load_model
#Test the trained model
def test_model():
    img = image.load_img("cat.1.jpg",target_size=(224,224))
    img = np.asarray(img)
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    saved_model = load_model("model_25-0.73.h5")
    output = saved_model.predict(img)
    if output[0][0] > output[0][1]:
        print("cat")
    else:
        print('dog')


if __name__ == '__main__':
    train_model()
    # test_model()

