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
n_feat = 20
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
# Cria uma coluna para cada gênero
genre_col = df['Genre'].str.split(',\s*', expand=True).stack().unique()
for col in genre_col:
    df[col] = df['Genre'].str.contains(col)
# %%
# Cria uma coluna para cada língua
lang_col = df['Languages'].str.split(',\s*', expand=True).stack().unique()
for col in lang_col:
    df[col] = df['Languages'].str.contains(col)
# %%
# Separa filmes de séries
df['Series or Movie'] = df['Series or Movie'].str.contains('Movie').astype(int)
#%%
# Dividir conjuntos
df = df[df['IMDb Score'].notna()]
# L: Teste só com colunas numéricas atuais
image_col = ['Image Feat '+str(x) for x in range(n_feat)]
# num_col = np.concatenate((np.concatenate((genre_col,image_col)),['Movie', 'Series','Awards Received', 'Awards Nominated For']))
num_col = np.concatenate((lang_col, genre_col, image_col, ['Series or Movie','Awards Received', 'Awards Nominated For']))
all_col = np.concatenate((num_col,['IMDb Score']))
df = df[all_col]
df = df.dropna()
df[genre_col] = df[genre_col].astype(int)
df[lang_col] = df[lang_col].astype(int)
corr = df.corr()['IMDb Score'].sort_values(key=lambda x: abs(x)).dropna()
top_col = num_col[-60:]
y = df['IMDb Score'].to_numpy()
x = df[top_col]
x = x.iloc[:,:].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.3,
    shuffle= True,
    random_state=1
    )


# %%
# Define modelo
# L: Modelo simples, só pra ver se tá rodando
def build_model():
  model = keras.Sequential([
    Dense(32, activation='relu', input_shape=[x_train.shape[-1],]),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
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
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#%%
# Treina modelo
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(patience=64)
EPOCHS = 300
BATCH_SIZE = 16
model = build_model()

history = model.fit(
  x_train, y_train,
  validation_split=0.1,
  epochs=EPOCHS, 
  batch_size=BATCH_SIZE,
  verbose=0,
  callbacks=early_stop
  )

result = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
print(result)
# %%
# Treinamento
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Treinamento')
plt.ylabel('Erro médio absoluto')
plt.xlabel('Época')
plt.legend(['treino', 'validação'], loc='upper right')
plt.show()
# %%
# Gráfico do erro
y_pred = model.predict(x_test)
samples = np.arange(len(y_pred))
y_error = [abs(x - y) for x, y in zip(y_pred,y_test)]
y_error =  [err for sub in y_error for err in sub]
y_error.sort()
sns.displot(y_error)
plt.xlabel('Erro absoluto')
plt.ylabel('Contagem')
plt.show()
#%%
# Vizualização das previsões
plt.scatter(y_test,y_pred, alpha=.1, s=4)
plt.plot([0,10],[0,10],c='r')
plt.ylabel('Valor previsto')
plt.xlabel('Valor real')
plt.show()
# %%
