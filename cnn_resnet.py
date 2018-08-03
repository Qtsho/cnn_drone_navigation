
# Importing the Keras libraries and packages
from keras.applications.resnet50 import ResNet50
from keras.models import model_from_json ,Model  
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, GlobalAveragePooling2D
nb_train_samples = 5839
nb_validation_samples = 1419
epochs = 47
batch_size =3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#def gen_with_norm(gen, normalize):
#    for x, y in gen:
#        yield normalize(x), y



base_model = ResNet50(include_top = False, weights = 'imagenet', input_shape =(197,197,3))
for layer in base_model.layers: 
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
#adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics =['accuracy'])


print model.summary()
#model.add(Dense(6, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(6, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))



# Part 2 - Fitting the CNN to the images


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
      )

test_datagen = ImageDataGenerator(rescale=1./255,
                                 )

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(197, 197),
        batch_size=batch_size,
        shuffle=True,
        class_mode= 'binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(197, 197),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')

#train_datagen.fit(training_set)
#test_datagen.fit(training_set)
model.fit_generator(
 training_set,
        steps_per_epoch=nb_train_samples,
        epochs=epochs,
        shuffle=True,
        validation_data = test_set,
        validation_steps=nb_validation_samples,
              verbose =1 )
              
#from keras.models import model_from_json   
model_json = model.to_json()
with open("model_CNNresnet_197.json", "w") as json_file:
   json_file.write(model_json)
# serialize weights to HDF5
        
        
model.save_weights("weights_CNNresnet_197.h5")
print("Saved model to disk")

json_file = open('model_CNNresnet_197.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("weights_CNNresnet_197.h5")
print("Loaded model from disk")


import numpy as np
from keras.preprocessing.image import image

test_image = image.load_img('./dataset/single_prediction/sigle1.png',target_size= (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
label = training_set.class_indices
prediction = classifier.predict(test_image)
if prediction[0][0] == 0:
    print 'This space is occupied'
else:
        print 'Thís is free space'


