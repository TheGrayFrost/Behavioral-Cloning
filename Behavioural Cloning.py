#!/usr/bin/env python
# coding: utf-8

# In[19]:


import cv2
import csv
from scipy import ndimage
import numpy as np


# In[10]:


lines = []
with open("./data/data/driving_log.csv") as dl:
    reader = csv.reader(dl)
    for line in reader:
        lines.append(line)


# In[15]:


lines = lines[1:]


# In[30]:


image.shape


# In[17]:


source_path = "./data/data/"
images = []
measurements = []
for line in lines:
    image = ndimage.imread(source_path + line[0])
    images.append(image)
    measurements.append(float(line[3]))


# In[20]:


X_train = np.array(images)
y_train = np.array(measurements)


# In[25]:


import tensorflow as tf


# In[29]:


from keras.models import Sequential
from keras.layers import Dense, Flatten


# In[37]:


model = Sequential()
model.add(Flatten(input_shape = (160,320,3)))
model.add(Dense(1))


# In[38]:


model.compile(optimizer='adam', loss = 'mse')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, epochs=6)


# In[40]:


model.save('model.h5')


# In[ ]:




