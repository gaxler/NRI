
# coding: utf-8

# In[1]:


import urllib

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


# In[2]:


from trains import Task
task = Task.init(project_name="examples", task_name='circles_and_mnist')
task.get_logger().set_default_upload_destination('s3://allegro-tutorials/ComputerVision')


# In[3]:


task_params = {
    # Plots
    'num_scatter_samples': 60, 'sin_max_value': 20, 'sin_steps': 30,
    # Mnist MLP
    'batch_size': 128, 'nb_classes': 10, 'nb_epoch': 6, 'hidden_dim':512
}
task_params = task.connect(task_params)


# In[4]:


task_params['custom_message'] = 'Hey! I can add params to a dict and they will be logged automagically!!'
print('Checkout the custom message:')
print('*'*len(task_params['custom_message']))
print(task_params['custom_message'])
print('*'*len(task_params['custom_message']))


# In[5]:


N = task_params['num_scatter_samples']
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (50 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Nice Circles')
plt.show()

x = np.linspace(0, task_params['sin_max_value'], task_params['sin_steps'])
y = np.sin(x)
plt.plot(x, y, 'o', color='black')
plt.title('Sinus Dots')
plt.show()


# In[6]:


import numpy as np
m = np.eye(256, 256, dtype=np.uint8)
plt.imshow(m)
plt.title('test2');


# In[7]:


def url_to_image(url):
  resp = urllib.request.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  return image

m = url_to_image('https://raw.githubusercontent.com/gaxler/NRI/master/nri.png')
task.get_logger().report_image_and_upload("debug", "we-can-log-images", iteration=1, matrix=m)


# ### Train an MNIST MLP

# In[8]:


batch_size = task_params['batch_size']
nb_classes = task_params['nb_classes']
nb_epoch = task_params['nb_epoch']


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

hidden_dim = task_params['hidden_dim']
model = Sequential()
model.add(Dense(hidden_dim, input_shape=(784,)))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(hidden_dim))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

board = TensorBoard(histogram_freq=1, log_dir='/tmp/histogram_example')
model_store = ModelCheckpoint(filepath='./weight.{epoch}.hdf5')


history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    callbacks=[board, model_store],
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[9]:


task.get_logger().flush()
task.close()
print('We are done :)')

