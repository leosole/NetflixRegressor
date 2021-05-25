#%%
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

from keras.applications.vgg16 import VGG16 
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

df = pd.read_csv('netflix-rotten-tomatoes-metacritic-imdb.csv')

path = r"/Users/leo/netflix/img"
os.chdir(path)

images = []
with os.scandir(path) as files:
    for filename in files:
        if filename.name.endswith('.jpg'):
            images.append(filename.name)
       
            
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

for image in images:
    try:
        feat = extract_features(image,model)
        data[image] = feat
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)
          
 # %%
filenames = np.array(list(data.keys()))

feat = np.array(list(data.values()))

feat = feat.reshape(-1,4096)

pca = PCA(n_components=10, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

kmeans = KMeans(n_clusters=4, random_state=22)
kmeans.fit(x)

groups = {}
for filename, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(filename)
    else:
        groups[cluster].append(filename)

# function that lets you view a cluster (based on identifier)        
def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
        
   
# this is just incase you want to see which value for k might be the best 
sse = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');
# %%
