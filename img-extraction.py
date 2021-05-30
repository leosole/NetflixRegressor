#%%
# Importar bibliotecas
import numpy as np 
import pandas as pd 
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.layers import Dense

from sklearn.decomposition import PCA

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#%%
# Número de colunas de parâmetros de imagem
n_feat = 4
#%%
# Carregar dados
path = r"/Users/leo/netflix" # mudar para caminho certo
os.chdir(path)
df = pd.read_csv('netflix-rotten-tomatoes-metacritic-imdb.csv')
df = df.sample(frac=1, random_state=1 )

# Listar imagens
path = r"/Users/leo/netflix/img" # mudar para caminho certo
os.chdir(path)

images = []
with os.scandir(path) as files:
    for f in files:
        if f.name.endswith('.jpg'):
            images.append(f.name)

#%%
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
p = r"/Users/leo/netflix/fvector" # mudar para caminho certo

# Nomes das colunas
img_list = df['Image'].to_list()
for feat in range(n_feat):
    df['Image Feat '+str(feat)] = np.nan

#%%
# Extrair features
for index, img in enumerate(img_list):
    if img != np.nan:
        name = img.split("/")[-1].split('?')[0]
        if name in images:
            try:
                feat = extract_features(name,model)
                data[img] = feat
            except:
                with open(p,'wb') as f:
                    pickle.dump(data,f)

#%%
# Diminuir a dimensão para n_feat
feat = np.array(list(data.values()))
feat = feat.reshape(-1,4096)

pca = PCA(n_components=n_feat, random_state=1)
pca.fit(feat)
img_feat = pca.transform(feat)
#%%
# Preencher colunas
for i, key in enumerate(data):
    data[key] = img_feat[i]

for row, feats in data.items():
    for f in range(n_feat):
        df.loc[df.Image == row, 'Image Feat '+str(f) ] = feats[f]

#%%
# Salva tabela
path = r"/Users/leo/netflix" # mudar para caminho certo
os.chdir(path)
df.to_csv('netflix-rotten-tomatoes-metacritic-imdb.csv')
# %%
