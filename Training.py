import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import datasets
iris=datasets.load_iris()
X=iris.data
y=iris.target
import pandas as pd
df_features= pd.DataFrame(X)
df_features
df_target= pd.DataFrame(y)
df_target

from sklearn.model_selection import train_test_split
#split train test dataset
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

#report folder
if (not(os.path.isdir("report"))):
    os.mkdir("report")

# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(4, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(1, name="layer3", activation="sigmoid"),
    ]
)

#compiler model
loss_fn = tf.keras.losses.mean_squared_error
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=keras.optimizers.get(opt), loss= loss_fn)
history = model.fit(X_train, y_train, epochs=50, verbose=2)

# printing summary
fstructure = open("report/structure.txt", "w")
model.summary(print_fn=lambda x: fstructure.write(x + '\n'))
fstructure.close()

# evaluation
losses = history.history['loss']
epochs = history.epoch

plt.plot(epochs, losses)
plt.xlabel("epochs")
plt.ylabel('losses')
plt.title('Loss per Epoch')
plt.show()
plt.savefig('report/history.png')
fmetrics = open("report/metrics.txt", "w")
fmetrics.write('Initial loss value : '+ str(losses[0]) + "\n")
fmetrics.write('Final loss value : ' + str(losses[-1]))
fmetrics.close()

# saving
model.save('report/weights.h5')

