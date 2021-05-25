#%%
import shutil # to save it locally
import pandas as pd

df = pd.read_csv('netflix-rotten-tomatoes-metacritic-imdb.csv')
df.shape
#%%
notfound = 0
for index, row in df.iterrows():
    image_url = row['Image']
    filename = image_url.split("/")[-1].split('?')[0]
    if row['IMDb Score'] < 5:
        try:
            shutil.move('img/'+filename, 'img/0-5/'+filename)
        except:
            notfound += 1
    elif row['IMDb Score'] < 6:
        try:
            shutil.move('img/'+filename, 'img/5-6/'+filename)
        except:
            notfound += 1
    elif row['IMDb Score'] < 7:
        try:
            shutil.move('img/'+filename, 'img/6-7/'+filename)
        except:
            notfound += 1
    elif row['IMDb Score'] < 8:
        try:
            shutil.move('img/'+filename, 'img/7-8/'+filename)
        except:
            notfound += 1
    elif row['IMDb Score'] < 11:
        try:
            shutil.move('img/'+filename, 'img/8-10/'+filename)
        except:
            notfound += 1
    else:
        try:
            shutil.move('img/'+filename, 'img/none/'+filename)
        except:
            notfound += 1
print(notfound)
# %%
