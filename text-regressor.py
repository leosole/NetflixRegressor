# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

batch_size = 32
epochs = 5
max_features = 10000
embedding_dim = 32
sequence_length = 250

df = pd.read_csv('netflix-rotten-tomatoes-metacritic-imdb.csv')
# %%
df['target'] = np.where(df['IMDb Score'] > 7, 1, 0)

# %%
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers.experimental.preprocessing import TextVectorization
txt = df.loc[:,['Summary', 'target', 'IMDb Score']]
txt['Summary'] = df['Summary'].fillna('null')
txt.sample(frac=1, random_state=10)
txt = txt.dropna()
#%%
sentences_train, sentences_test, y_train, y_test = train_test_split(
    txt['Summary'], txt['target'], test_size=0.3, random_state=1000)
sentences_val, sentences_test, y_val, y_test = train_test_split(
    sentences_test, y_test, test_size=0.5, random_state=1000)
train_score, test_score, yy, yu = train_test_split(
    txt['IMDb Score'], txt['target'], test_size=0.3, random_state=1000)
val_score, test_score, vy, uu = train_test_split(
    test_score, yu, test_size=0.5, random_state=1000)

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_val = np.asarray(y_val).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
test_score = np.asarray(test_score).astype('float32').reshape((-1,1))

train = tf.data.Dataset.from_tensor_slices((sentences_train.values, y_train))
val = tf.data.Dataset.from_tensor_slices((sentences_val, y_val))
test = tf.data.Dataset.from_tensor_slices((sentences_test, y_test))

#%%
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

text_ds = train.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

# Vectorize the data.
x_train = train.map(vectorize_text)
x_val = val.map(vectorize_text)
x_test = test.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
x_train = x_train.cache().prefetch(buffer_size=10)
x_val = x_val.cache().prefetch(buffer_size=10)
x_test = x_test.cache().prefetch(buffer_size=10)

#%%
from tensorflow.keras import layers

# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(56, activation="relu")(x)
x = layers.Dropout(0.3)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#%%

# Fit the model using the train and test datasets.
model.fit(x_train, validation_data=x_val, epochs=epochs)
model.evaluate(x_test)
y_pred = model.predict(x_test)

import matplotlib.pyplot as plt
plt.scatter(y_pred,test_score)
# %%
