# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dropout,BatchNormalization
from keras.layers import Dense


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Initialising the CNN
classifier = Sequential ()
# Step 1 - Convolution
classifier.add(Convolution2D(filters = 96, kernel_size= [3,3],activation = 'relu',input_shape=(128, 128, 3)))
# Step 2 - Pooling
classifier.add (MaxPooling2D(pool_size= (2,2), strides = 2))
classifier.add(BatchNormalization(axis=3))
classifier.add(Convolution2D(filters = 256, kernel_size= [3,3],activation = 'relu'))
classifier.add (MaxPooling2D(pool_size= (2,2), strides = 2))
classifier.add(BatchNormalization(axis=3))

classifier.add(Convolution2D(filters = 384,strides= 1, kernel_size = [3,3],activation = 'relu'))
classifier.add(Convolution2D(filters = 384,strides= 1, kernel_size = [3,3],activation = 'relu'))
classifier.add(Convolution2D(filters = 256,strides= 1, kernel_size = [3,3],activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3),strides = 2))
classifier.add(Dropout(0.5))
classifier.add(BatchNormalization(axis=3))


#Step 3 - Flattening
classifier.add (Flatten())
#Step 4 - Full Connection
classifier.add(Dense(128, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(128, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))
#Compile the CNN, schocastic gradient densense, binary entropy 
classifier.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics =['accuracy'])

print classifier.summary()
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=3,
        shuffle=True,
        class_mode= 'binary')
result= training_set.class_indices
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=3,
        shuffle=True,
        class_mode='binary')
        
classifier.fit_generator(
        training_set,
        shuffle=True,
        steps_per_epoch=5161,
        epochs=50,
        validation_data = test_set,
        validation_steps=1000,
         verbose =1 )
     

model_json = classifier.to_json()
with open("model_CNNAlexnet_128.json", "w") as json_file:
   json_file.write(model_json)
# serialize weights to HDF5
        
        
classifier.save_weights("weights_CNNAlexnet_128.h5")
print("Saved model to disk")


#from keras.models import model_from_json   
#json_file = open('model_CNNAlexnet_128.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#classifier = model_from_json(loaded_model_json)
## load weights into new model
#classifier.load_weights("weights_CNNAlexnet_128.h5")
#print("Loaded model from disk")



import numpy as np
from keras.preprocessing.image import image

test_image = image.load_img('dataset/single_prediction/sigle25.png',target_size= (128,128))
test_image = image.img_to_array(test_image)
test_image = test_image/255
test_image = np.expand_dims(test_image, axis = 0)
print classifier.predict(test_image)
prediction = classifier.predict(test_image)
if prediction[0][0] < 0.5:
    print 'This space is ocupied'
else:
        print 'This is free space'


