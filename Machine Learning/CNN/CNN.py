import numpy as np
import pandas as pd
from scipy import misc
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_normal
from keras.utils import np_utils

batch_size = 128
epochs = 20
BATCH_NORM = True


train_directory = os.path.join('D:\Adithya\Dataset\data','train')
labels = os.listdir(train_directory)
label_dict = dict()
i=0
for label in labels:
    label_dict[label] = i
    i = i+1

meta_data = 'D:\Adithya\Dataset\Codes\Meta_Data.csv'
meta_df = pd.read_csv(meta_data, encoding='utf-8')

indices = list(meta_df)
ytrain = meta_df['label']
y1 = []
for y in ytrain:
    y1.append(label_dict[y])
ytrain = y1
xtrain = []
y_train = []


for i in range(0,len(meta_df['filepath'])):
    if(misc.imread(meta_df['filepath'][i]).shape ==  (64,64,3)):
        xtrain.append(misc.imread(meta_df['filepath'][i]))
        y_train.append(ytrain[i])
        print('Read[TRA] : '+str(i)+'th Image, dimension  :'+str(misc.imread(meta_df['filepath'][i]).shape))
xtrain = np.stack(xtrain, axis=0)
ytrain = np.array(y_train)
print('xtrain shape :')
print(xtrain.shape)
print('ytrain shape :')
print(ytrain.shape)

test_base_path = 'D:\Adithya\Dataset\data\\val\images'
test_df = pd.read_csv('D:\Adithya\Dataset\data\\val\\val_annotations.txt',
                      delimiter='\t', names=['fname', 'label','x_min', 'y_min', 'x_max', 'y_max'])
ytest = test_df['label']
y2 = []
for y in ytest:
    y2.append(label_dict[y])
ytest = y2

xtest = []
y_test = []
for i in range(0,len(test_df['fname'])):
    val_image = os.path.join(test_base_path,test_df['fname'][i])
    if misc.imread(val_image).shape == (64,64,3):
        xtest.append(misc.imread(val_image))
        y_test.append(ytest[i])
        print('Read[VAL] : '+str(i)+'th Image')
xtest = np.stack(xtest, axis=0)
ytest = np.array(y_test)

print('xtest shape :')
print(xtest.shape)
print('ytest shape :') 
print(ytest.shape)



num_classes = len(labels)

ytrain = np_utils.to_categorical(ytrain, num_classes)
ytest = np_utils.to_categorical(ytest, num_classes)
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain  /= 255
xtest /= 255


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=xtrain.shape[1:], name='block1_conv1'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

#model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

#model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='same', name='block2_conv2'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))


model.add(AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool'))

#model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))

#model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))

#model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), padding='same', name='block3_conv4'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

#model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))

#model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))

#model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))

#model.add(Conv2D(256, (3, 3), padding='same', name='block4_conv4'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))
#model.add(AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool'))

#model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))

#model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))

#model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))
#
model.add(Conv2D(256, (3, 3), padding='same', name='block5_conv4'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(2048, name='fc1'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512, name='fc2'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))
model.add(Dropout(0.4))

#model.add(Dense(256, name='fc3'))
#model.add(BatchNormalization()) if BATCH_NORM else None
#model.add(Activation('relu'))
#model.add(Dropout(0.6))

model.add(Dense(num_classes))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



model.summary()

cnn = model.fit(xtrain,ytrain, batch_size=batch_size, epochs=epochs,validation_data=(xtest,ytest),shuffle=True)

plt.figure(0)
plt.plot(cnn.history['acc'],'r')
plt.plot(cnn.history['val_acc'],'g')
plt.xticks(np.arange(0, 11, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])

plt.figure(1)
plt.plot(cnn.history['loss'],'r')
plt.plot(cnn.history['val_loss'],'g')
plt.xticks(np.arange(0, 11, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])
plt.show()
 