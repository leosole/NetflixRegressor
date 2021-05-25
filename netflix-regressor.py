#%%
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

from keras.applications.vgg16 import VGG16 
from keras.models import Model

from sklearn.decomposition import PCA

import os
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('netflix-rotten-tomatoes-metacritic-imdb.csv')
df = df.sample(frac=1, random_state=1 )

# Remover colunas desnecessárias
df.drop (
    axis = 'columns', 
    columns = ['Netflix Link', 'IMDb Link', 'Summary', 'Poster', 'TMDb Trailer', 'Boxoffice', 'Hidden Gem Score', 'View Rating', 'Rotten Tomatoes Score', 'Metacritic Score', 'IMDb Votes', 'Trailer Site'], 
    inplace = True)

# Preencher nan com 0
df['Awards Received'] = df['Awards Received'].fillna(0)
df['Awards Nominated For'] = df['Awards Nominated For'].fillna(0)

df = df[df['IMDb Score'].notna()]
y = df['IMDb Score'].to_numpy()
x = df.drop(columns = ['IMDb Score'])
x = x.iloc[:,:].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.3,
    shuffle= True,
    random_state=1
    )

# %%
path = r"/Users/leo/netflix/img"
os.chdir(path)

# Lista com imagens
images = []
with os.scandir(path) as files:
    for f in files:
        if f.name.endswith('.jpg'):
            images.append(f.name)

# Modelo para extração de features    
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(imagefile, model):
    img = load_img(imagefile, target_size=(224,224))
    img = np.array(img) 
    reshaped_img = img.reshape(1,224,224,3) 
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=True)
    return features
   
data = {}
p = r"/Users/leo/netflix/fvector"

# FAZER lógica para selecionar somente imagens do conjunto de treino
for image in images:
    try:
        feat = extract_features(image,model)
        data[image] = feat
    except:
        with open(p,'wb') as f:
            pickle.dump(data,f)

feat = np.array(list(data.values()))
feat = feat.reshape(-1,4096)
# Diminui a dimensão
pca = PCA(n_components=10, random_state=1)
pca.fit(feat)
x = pca.transform(feat)