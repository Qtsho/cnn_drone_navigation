# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_model = VGG16(weights='imagenet', include_top=False,input_shape =(128,128,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics =['accuracy'])
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

    
print model.summary()

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

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=3,
         shuffle=True,
        class_mode='binary')
        
model.fit_generator(
        training_set,
        steps_per_epoch=5839,
        epochs=50,
        validation_data = test_set,
        validation_steps=1419,
        verbose =1 
       )
     
#from keras.models import model_from_json   
model_json = model.to_json()
with open("weights_CNNVGG16_128.json", "w") as json_file:
   json_file.write(model_json)
# serialize weights to HDF5
        
        
model.save_weights("weights_CNNVGG16_128.h5")
print("Saved model to disk")

#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#classifier = model_from_json(loaded_model_json)
## load weights into new model
#classifier.load_weights("model.h5")
#print("Loaded model from disk")
# respect this shape of an image


import numpy as np
from keras.preprocessing.image import image

test_image = image.load_img('./dataset/single_prediction/crash2.png',target_size= (128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
label = training_set.class_indices
prediction = model.predict(test_image)
if prediction[0][0] == 0:
    print 'This space is occupied'
else:
        print 'Thís is free space'


