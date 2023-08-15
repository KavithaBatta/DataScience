#!/usr/bin/env python
# coding: utf-8

# # cnn happy and sad

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
import cv2
import os


# In[ ]:





# In[1]:


import numpy as np


# In[ ]:





# In[2]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


from tensorflow.keras.preprocessing import image


# In[4]:


import matplotlib.pylab as plt


# In[5]:


import tensorflow as tf


# In[6]:


import numpy as np


# In[7]:


import cv2


# In[8]:


import os


# In[21]:


img=image.load_img(r'C:\Users\Admin\Desktop\NIT Data Science Course\19. CNN - Happy  or Sad 1st aug\19. CNN - Happy  or Sad\training\happy\1.jpg')


# In[22]:


img


# In[23]:


plt.imshow(img)


# In[26]:


i1= cv2.imread(r'C:\Users\Admin\Desktop\NIT Data Science Course\19. CNN - Happy  or Sad 1st aug\19. CNN - Happy  or Sad\training\happy\1.jpg')


# In[27]:


i1


# In[28]:


i1.shape


# In[29]:


train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)


# In[30]:


train_dataset=train.flow_from_directory(r'C:\Users\Admin\Desktop\NIT Data Science Course\19. CNN - Happy  or Sad 1st aug\19. CNN - Happy  or Sad\training',
                                       target_size=(200,200),
                                        batch_size=3,
                                        class_mode='binary')
validation_dataset=validation.flow_from_directory(r'C:\Users\Admin\Desktop\NIT Data Science Course\19. CNN - Happy  or Sad 1st aug\19. CNN - Happy  or Sad\validation',
                                                 target_size=(200,200),
                                                  batch_size=3,
                                                  class_mode='binary')


# In[31]:


train_dataset.class_indices


# In[33]:


train_dataset.classes


# In[34]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()


# In[56]:


model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.legacy.RMSprop(lr=0.001),
              metrics=['acc'])


# In[41]:


model_fit=model.fit(train_dataset,
                   steps_per_epoch=3,
                   epochs=10,
                   validation_data=validation_dataset) 


# In[59]:


get_ipython().run_line_magic('pinfo2', 'history.history')
import keras
from matplotlib import pyplot as plt
#history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
history=model_fit=model.fit(train_dataset,steps_per_epoch=3,epochs=10,validation_data=validation_dataset) 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[60]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:




