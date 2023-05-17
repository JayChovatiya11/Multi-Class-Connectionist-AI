#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing


# In[9]:


# setting path for animal dataset
dataset_path = "C:\\Users\\dell\\Desktop\\animal\\raw-img\\"
# Load the dataset
x = []
y = []
for i in os.listdir(dataset_path):
    animaldir = os.path.join(dataset_path, i)
    if os.path.isdir(animaldir):
        for file_name in os.listdir(animaldir):
            file_path = os.path.join(animaldir, file_name)
            image = keras.preprocessing.image.load_img(file_path, target_size=(32, 32))
            image_array = keras.preprocessing.image.img_to_array(image)
            x.append(image_array)
            y.append(i)
x = np.array(x)
y = np.array(y)


# In[3]:


# Separating the into training and test sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)


# In[4]:


# Creating the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(np.unique(y)), activation="softmax")
])
#Adding Adam optimizer in the model 
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# In[5]:


# Convert the labels to one-hot encoding
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = keras.utils.to_categorical(y_train, len(np.unique(y)))
y_test = keras.utils.to_categorical(y_test, len(np.unique(y)))


# In[6]:


# Training the model and seeting epochs to 2 
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

#Evaluate the model
_, acc = model.evaluate(x_test, y_test, verbose=0)
print("Model accuracy is : {:.2f}%".format(acc * 100))


# In[12]:


# OPTIONAL : hypertuning the model to increase the accuracy 
#model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.02, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
# Train the model
#model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))
model.save("C:\\Users\\dell\\Desktop\\animal\\")
model = keras.models.load_model("C:\\Users\\dell\\Desktop\\animal\\")


# In[14]:


# classify image of elephant
image = keras.preprocessing.image.load_img("C:\\Users\\dell\\Desktop\\animal\\raw-img\\elefante\\elephant.jpg", target_size=(32, 32))
image_array = keras.preprocessing.image.img_to_array(image)
predictions = model.predict(image_array[np.newaxis, ...])
predicted_class = label_encoder.inverse_transform(np.array([np.argmax(predictions)]))
print("Your Predicted class is:", predicted_class)


# In[ ]:




