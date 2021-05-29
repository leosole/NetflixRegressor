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
import pickle

#%%
# Carregar dados
path = r"/Users/leo/netflix" # mudar para caminho certo
os.chdir(path)
df = pd.read_csv('netflix-rotten-tomatoes-metacritic-imdb.csv')
df = df.sample(frac=1, random_state=1 )
#%%
# Remover colunas desnecessárias
df.drop (
    axis = 'columns', 
    columns = ['Netflix Link', 'IMDb Link', 'Summary', 'Poster', 'TMDb Trailer', 'Boxoffice', 'Hidden Gem Score', 'View Rating', 'Rotten Tomatoes Score', 'Metacritic Score', 'IMDb Votes', 'Trailer Site'], 
    inplace = True)

# Preencher nan com 0
df['Awards Received'] = df['Awards Received'].fillna(0)
df['Awards Nominated For'] = df['Awards Nominated For'].fillna(0)
# %%
# Cria uma oluna para cada gênero
genre_col = df['Genre'].str.split(',\s*', expand=True).stack().unique()
for col in genre_col:
    df[col] = df['Genre'].str.contains(col)
# %%
# Listar imagens
path = r"/Users/leo/netflix/img" # mudar para caminho certo
os.chdir(path)

images = []
with os.scandir(path) as files:
    for f in files:
        if f.name.endswith('.jpg'):
            images.append(f.name)

#%%
# Dividir conjuntos
df = df[df['IMDb Score'].notna()]
y = df['IMDb Score'].to_numpy()
x = df.drop(columns = ['IMDb Score'])
# L: Teste só com colunas numéricas atuais
num_col = np.concatenate((genre_col,['Awards Received', 'Awards Nominated For', 'Image']))
df = df.dropna()
x = df[num_col]
x = x.iloc[:,:].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.3,
    shuffle= True,
    random_state=1
    )


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
data_test = {}
p = r"/Users/leo/netflix/fvector" # mudar para caminho certo

img_list_train = [x[-1] for x in x_train]
img_list_test = [x[-1] for x in x_test]

#%%
# Separar imagens de treino e teste
# Lista para guardar erros
bad_train_ind = []
bad_test_ind = []

for index, img in enumerate(img_list_train):
    name = img.split("/")[-1].split('?')[0]
    if name in images:
        try:
            feat = extract_features(name,model)
            data[name] = feat
        except:
            bad_train_ind.append(index)
            with open(p,'wb') as f:
                pickle.dump(data,f)
    else:
        bad_train_ind.append(index)

for index, img in enumerate(img_list_test):
    name = img.split("/")[-1].split('?')[0]
    if name in images:
        try:
            feat = extract_features(name,model)
            data_test[name] = feat
        except:
            bad_test_ind.append(index)
            with open(p,'wb') as f:
                pickle.dump(data,f)
    else:
        bad_test_ind.append(index)
#%%
# Remover erros do conjunto de teste
y_train = np.delete(y_train,bad_train_ind, 0)
y_test = np.delete(y_test,bad_test_ind, 0)

feat = np.array(list(data.values()))
feat = feat.reshape(-1,4096)
feat_test = np.array(list(data_test.values()))
feat_test = feat_test.reshape(-1,4096)

# Diminuir a dimensão para n_components
pca = PCA(n_components=5, random_state=1)
pca.fit(feat)
img_train = pca.transform(feat)
img_test = pca.transform(feat_test)

# %%
# Concatena parâmetros da tabela com parâmetros das imagens
X_train = [np.append(row[:-1], img) for row, img in zip(x_train, img_train)]
X_test = [np.append(row[:-1], img) for row, img in zip(x_test, img_test)]
# %%
# Define modelo
# L: Modelo simples, só pra ver se tá rodando
def build_model():
  model = keras.Sequential([
    Dense(64, activation='relu', input_shape=[X_train.shape[-1],]),
    Dense(64, activation='relu'),
    Dense(1)
  ])

  optimizer = keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
#%%
# Padroniza amostras
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
# Treina modelo
EPOCHS = 300
BATCH_SIZE = 64
model = build_model()

history = model.fit(
  X_train, y_train,
  epochs=EPOCHS, 
  batch_size=BATCH_SIZE,
  verbose=0,
  )

result = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
print(result)
# %%
y_pred = model.predict(X_test)
x_axis = np.arange(len(y_pred))
y_error = [abs(x - y) for x, y in zip(y_pred,y_test)]
y_error.sort()
plt.scatter(x_axis,y_error, label='Erro Absoluto')
plt.legend()

# %%
