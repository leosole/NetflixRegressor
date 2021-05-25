import shutil # to save it locally
import os
from random import sample

error = 0
classes = ['0-5','5-6','6-7','7-8','8-10','none']
for c in classes:
    imgs = os.listdir('img/train/'+c)
    val = sample(imgs,int(len(imgs)*3/10))
    for filename in val:
        try:
            shutil.move('img/train/'+c+'/'+filename, 'img/validation/'+c+'/'+filename)
        except:
            error += 1
print(error)