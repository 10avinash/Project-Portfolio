import pandas as pd
import os
import glob

train_directory = os.path.join('D:\Adithya\Dataset\data','train')
labels = os.listdir(train_directory)
txt_files = glob.glob(os.path.join(os.path.join(train_directory,'*'),'*.txt'))
flag = 0
for txt_file in txt_files:
    if flag is 0:
        data = pd.read_csv(txt_file,delimiter='\t',names=['Image','x_min', 'y_min', 'x_max', 'y_max'])
        flag = 1
    if flag is not 0:
        data = data.append(pd.read_csv(txt_file,delimiter='\t',names=['Image','x_min', 'y_min', 'x_max', 'y_max']),ignore_index=True)
list_of_images = data['Image']
fpath = []
label = []
for image in list_of_images:
    fpath.append(os.path.join(os.path.join(os.path.join(train_directory,image.split('_')[0]),'images'),image))
    label.append(image.split('_')[0])

data = data.assign(filepath = fpath)
data = data.assign(label = label)
data.to_csv('Meta_Data.csv', encoding='utf-8', index=False)
