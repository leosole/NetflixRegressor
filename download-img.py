
#%%
# ## Importing Necessary Modules
import requests # to get image from the web
import shutil # to save it locally
import pandas as pd
from os.path import exists

df = pd.read_csv('netflix-rotten-tomatoes-metacritic-imdb.csv')
#%%
for index, row in df.iterrows():
    image_url = row['Image']
    filename = image_url.split("/")[-1]
    savename = image_url.split("/")[-1].split('?')[0]
    if exists('img/'+savename) == False:
        r = requests.get(image_url, stream = True)

        if r.status_code == 200:
            r.raw.decode_content = True
            
            with open(filename.split('?')[0],'wb') as f:
                shutil.copyfileobj(r.raw, f)
                
            print('Image sucessfully Downloaded: ',filename)
        else:
            print('Image Couldn\'t be retreived')
# %%
