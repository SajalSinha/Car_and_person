#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from tqdm import tqdm
from bboxannotations import *


# In[3]:


# import pandas library as pd
import pandas as pd
  
# create an Empty DataFrame
# object With column names only
column_names = ["file_name", "width", "height","id"]
df = pd.DataFrame(columns = column_names, dtype=object)
df


# In[9]:


# Python program to demonstrate
# Conversion of JSON data to
# dictionary
 
 
# importing the module
import json
 
# Opening JSON file
with open('bboxannotations.json') as json_file:
    data = json.load(json_file)
 
    # Print the type of data variable
    print("Type:", type(data))
 
    # Print the data of dictionary
    print("Images:", data['images'])
    print("\nAnnotations:", data['annotations'])


# In[10]:


for k, v in data.items():
    print(k)
    if k=='images':
        for ele in tqdm(v):
            #print(ele)
            df = df.append({'file_name' : ele['file_name'], 'width' : ele['width'], 'height' : ele['height'], "id": ele['id']}, 
                ignore_index = True)


# In[11]:


df.shape


# In[12]:


df


# In[13]:


column_names = ["category_id", "image_id", "bbox"]
df_annotations = pd.DataFrame(columns = column_names, dtype=object)
df_annotations


# In[14]:


for k, v in data.items():
    print(k)
    if k=='annotations':
        for ele in tqdm(v):
            #print(ele)
            df_annotations = df_annotations.append({'category_id' : ele['category_id'], 'image_id' : ele['image_id'], 'bbox' : ele['bbox']}, 
                ignore_index = True)


# In[15]:


df_annotations


# In[16]:


df_annotations["image_id"].nunique()


# In[17]:


df_1=pd.merge(df, df_annotations, left_on='id', right_on='image_id', how='left')


# In[18]:


df_1.shape


# In[19]:


df_1


# In[20]:


def process(row):
    lst=row['bbox']
    return ["{:.6f}".format(lst[0]/row['width']), "{:.6f}".format(lst[1]/row['height']),"{:.6f}".format(lst[2]/row['width']),"{:.6f}".format(lst[3]/row['height'])]
    
df_1['bbox_normalised']=df_1.apply(process, axis=1)


# In[21]:


from tqdm import tqdm
tqdm.pandas()


# In[22]:


df_1['cent_x_bbox_normalised']=df_1['bbox_normalised'].progress_apply(lambda x : x[0])
df_1['cent_y_bbox_normalised']=df_1['bbox_normalised'].progress_apply(lambda x : x[1])
df_1['width_bbox_normalised']=df_1['bbox_normalised'].progress_apply(lambda x : x[2])
df_1['height_bbox_normalised']=df_1['bbox_normalised'].progress_apply(lambda x : x[3])


# In[23]:


"{:.2f}".format(13.949999999999999)


# In[24]:


df_1


# In[26]:


df_1["category_id"].replace({2: 1, 1: 0}, inplace=True)


# In[28]:


for i in tqdm(range(df.shape[0])):
    #print(i)
    filename=df_1[df_1["image_id"]==i]["file_name"].iloc[0][:15] + '.txt'
    df_temp=df_1[df_1["image_id"]==i][["category_id",'cent_x_bbox_normalised', 'cent_y_bbox_normalised','width_bbox_normalised', 'height_bbox_normalised' ]]
    df_temp.to_csv(filename, sep=" ", index=False, header=False)


# In[ ]:




