#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.layers import Input
from keras.models import Model
from tensorflow.python.keras.models import Sequential 
from keras import regularizers
import tensorflow.python.keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import keras
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import (
    BatchNormalization, Activation, Flatten, Dropout, Dense
)

class HelperFunctions:
       
    def lr_schedule(self, epoch):
        lrate = 0.001
        if epoch > 75:
            lrate = 0.0005
        elif epoch > 100:
            lrate = 0.0003        
        return lrate
        
    def full_model(self, x_train, weight_decay, num_classes):    
        inputs1 = Input((32, 32, 3))
        x = Sequential()(inputs1)
        x = Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:])(x)
        x = Activation('elu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('elu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('elu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('elu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.3)(x)
        x = Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('elu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('elu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.4)(x)
        x = Flatten(name='client_output_layer')(x)      
        x = Dense(num_classes, activation = 'softmax')(x)           
        model = Model(inputs=inputs1, outputs=x)
        return model

    def client_model_create(self, i,  x_train, y_train, weight_decay, num_classes, batch_size, localepochs):
        
        train_data_x, test_data_x = np.split(x_train,[int(0.9 * len(x_train))])
        train_data_y, test_data_y = np.split(y_train,[int(0.9 * len(y_train))])

        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
            )
        datagen.fit(train_data_x)       
        
        fullmodel = self.full_model(train_data_x, weight_decay, num_classes)
        
        opt_rms = tensorflow.keras.optimizers.RMSprop(learning_rate=0.001,decay=1e-6)
        
        fullmodel.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
        history = fullmodel.fit(datagen.flow(train_data_x, train_data_y, batch_size=batch_size),\
                            steps_per_epoch=train_data_x.shape[0] // batch_size,epochs=localepochs,\
                            verbose=2,validation_data=(test_data_x,test_data_y), callbacks=[LearningRateScheduler(self.lr_schedule)])
        
        matplotlib.use('Agg')
        params = {'legend.fontsize': '20',
                  'figure.figsize': (5.5, 5),
                  'axes.labelsize': '20',
                  'axes.titlesize':'20',
                  'xtick.labelsize':'20',
                  'ytick.labelsize':'20'}
        plt.rcParams.update(params)
        
        # Plotting accuracy
        f1 = plt.figure()
        plt.plot(history.history['accuracy'],'b:',linewidth=3)
        plt.plot(history.history['val_accuracy'], 'r--',linewidth=3)
        plt.title('model accuracy', fontsize=20)
        plt.ylim((0.1,1.0))
        plt.ylabel('accuracy', fontsize=20)
        plt.xlabel('epoch', fontsize=20)
        plt.legend(['train', 'test'], loc='upper left') 
        #save plot
        figname = "client_no_"+str(i)+"_accuracy.pdf"
        f1.savefig(figname, bbox_inches='tight')
     
        # Plotting Loss
        f2 = plt.figure()
        plt.plot(history.history['loss'],'g-',linewidth=3)
        plt.plot(history.history['val_loss'],'m-.',linewidth=3)
        plt.title('model loss', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.xlabel('epoch', fontsize=20)
        plt.legend(['train', 'test'], loc='upper left')
        #save plot
        figname = "client_no_"+str(i)+"_loss.pdf"
        f2.savefig(figname, bbox_inches='tight')
        
        plt.close('all')  
        
        inter_output_model = keras.Model(fullmodel.input, fullmodel.get_layer('client_output_layer').output )  
        return inter_output_model
