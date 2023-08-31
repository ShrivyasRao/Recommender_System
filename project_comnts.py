#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all the dependencies
#from PIL import Image
import PIL.Image
import random
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
import time
import re
import os
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
from scipy.sparse import hstack
import plotly
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Layout

plotly.offline.init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")


# READ THE DATA

# In[2]:


data = pd.read_json('F:\\final year project\\phase 2\\work\\tops_fashion.json') #using Pandas to read the json file


# In[3]:


print ('Number of data points : ', data.shape[0],        'Number of features/variables:', data.shape[1])


# In[4]:


data.columns


# In[9]:


data = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name', 'title', 'formatted_price']] 
#storing the data of the mentioned fields


# In[10]:


print ('Number of data points : ', data.shape[0],        'Number of features:', data.shape[1])
data.head()


# In[7]:


print(data['product_type_name'].describe())


# In[8]:


print(data['product_type_name'].unique())


# In[9]:


product_type_count = Counter(list(data['product_type_name']))
product_type_count.most_common(10)


# In[10]:


print(data['brand'].describe())


# In[11]:


brand_count = Counter(list(data['brand']))
brand_count.most_common(10)


# In[12]:


print(data['color'].describe())


# In[13]:


color_count = Counter(list(data['color']))
color_count.most_common(10)


# In[14]:


print(data['formatted_price'].describe())


# In[15]:


price_count = Counter(list(data['formatted_price']))
price_count.most_common(10)


# In[16]:


print(data['title'].describe())


# In[17]:


#serialize object to file
data.to_pickle('F:\\final year project\\phase 2\\work\\pickels\\180k_apparel_data')


# In[18]:


data = data.loc[~data['formatted_price'].isnull()]
print('Number of data points After eliminating price=NULL :', data.shape[0])


# In[19]:


data =data.loc[~data['color'].isnull()]
print('Number of data points After eliminating color=NULL :', data.shape[0])


# In[20]:


data.to_pickle('F:\\final year project\\phase 2\\work\\pickels\\28k_apparel_data')


# In[3]:


data = pd.read_pickle('C:\\Users\\Shrivyas\\Desktop\\project\\all_proj_fold\\final_proj_data\\pickels\\28k_apparel_data')


# In[14]:


temp=list(data['asin'])
temp.index('B00AQ4GMTS')


# In[19]:


temp2=data['medium_image_url'].loc[data['asin']=='B00AQ4GN3I']


# In[20]:


for i,j in temp2.iteritems():
    print(j)


# In[22]:


#display head data
data.head()


# In[23]:


#sorting the data on the basis of title 
data_sorted = data[data['title'].apply(lambda x: len(x.split())>4)]
print("After removal of products with short description:", data_sorted.shape[0])


# In[24]:


data_sorted.sort_values('title',inplace=True, ascending=False)
data_sorted.head()


# In[25]:


indices = []
for i,row in data_sorted.iterrows():
    indices.append(i)


# In[26]:


#filtering duplicate entries
import itertools
stage1_dedupe_asins = []
i = 0
j = 0
num_data_points = data_sorted.shape[0]
while i < num_data_points and j < num_data_points:
    
    previous_i = i
    a = data['title'].loc[indices[i]].split()
    j = i+1
    while j < num_data_points:
        b = data['title'].loc[indices[j]].split()
        length = max(len(a), len(b))
        count  = 0
        for k in itertools.zip_longest(a,b): 
            if (k[0] == k[1]):
                count += 1
        if (length - count) > 2:
            stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[i]])
            i = j
            break
        else:
            j += 1
    if previous_i == i:
        break


# In[27]:


data = data.loc[data['asin'].isin(stage1_dedupe_asins)]


# In[28]:


print('Number of data points : ', data.shape[0])


# In[30]:


data.to_pickle('F:\\final year project\\phase 2\\work\\pickels\\17k_apperal_data')


# In[31]:



data = pd.read_pickle('F:\\final year project\\phase 2\\work\\pickels\\17k_apperal_data')


# In[32]:


#filtering duplicate entries(takes a lot of time)
indices = []
for i,row in data.iterrows():
    indices.append(i)

stage2_dedupe_asins = []
while len(indices)!=0:
    i = indices.pop()
    stage2_dedupe_asins.append(data['asin'].loc[i])
    a = data['title'].loc[i].split()
    for j in indices:
        
        b = data['title'].loc[j].split()
        length = max(len(a),len(b))
        count  = 0
        for k in itertools.zip_longest(a,b): 
            if (k[0]==k[1]):
                count += 1
        if (length - count) < 3:
            indices.remove(j)


# In[33]:


data = data.loc[data['asin'].isin(stage2_dedupe_asins)]


# In[34]:


print('Number of data points after stage two of dedupe: ',data.shape[0])


# In[ ]:


#data = pd.read_pickle('C:\\Users\\Vineeth Bekal\\Downloads\\PICKLE\\17k_apperal_data')


# In[35]:


data.to_pickle('F:\\final year project\\phase 2\\work\\pickels\\16k_apperal_data')


# In[36]:


#data.to_pickle('C:\\Users\\Vineeth Bekal\\Downloads\\PICKLE\\16k_apperal_data')
data = pd.read_pickle('F:\\final year project\\phase 2\\work\\pickels\\16k_apperal_data')


# In[135]:


#listing the stopwords
stop_words = set(stopwords.words('english')) 
#words which are filtered out before or after processing of natural language data
print ('list of stop words:', stop_words)

#function that adds stopwords filtered text back to our data
def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            word = ("".join(e for e in words if e.isalnum()))
            word = word.lower()
            if not word in stop_words:
                string += word + " "
        data[column][index] = string


# In[41]:


start_time = time.clock()
for index, row in data.iterrows():
    nlp_preprocessing(row['title'], index, 'title')
print(time.clock() - start_time, "seconds")


# In[42]:


data.head()


# In[ ]:


data.to_pickle('F:\\final year project\\phase 2\\work\\pickels\\16k_apperal_data_preprocessed')


# In[93]:


data = pd.read_pickle('F:\\final year project\\phase 2\\work\\pickels\\16k_apperal_data_preprocessed')
data.head()


# In[94]:


#reading and displaying the image here
def display_img(url,ax,fig):
    try:
        response = requests.get(url)
        img = PIL.Image.open(BytesIO(response.content))
        plt.imshow(img)
    except:
        pass
    
#Plotting a heatmap here
def plot_heatmap(keys, values, labels, url, text): #heatmap is a data visualization technique that shows magnitude of a phenomenon as color in two dimensions
    gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[4,1]) 
    fig = plt.figure(figsize=(25,3))
    ax = plt.subplot(gs[0])
    ax = sns.heatmap(np.array([values]), annot=np.array([labels]))
    ax.set_xticklabels(keys) 
    ax.set_title(text) 
    ax = plt.subplot(gs[1])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    display_img(url, ax, fig)
    plt.show()

#plotting the heatmap representations of the images
def plot_heatmap_image(doc_id, vec1, vec2, url, text, model):
    intersection = set(vec1.keys()) & set(vec2.keys()) 
    #print(vec1)
    #print(vec2)
    #print(intersection)
    for i in vec2:
        if i not in intersection:
            vec2[i]=0
    keys = list(vec2.keys())
    values = [vec2[x] for x in vec2.keys()]
    if model == 'bag_of_words':
        labels = values
    elif model == 'tfidf':
        labels = []
        for x in vec2.keys():
            if x in  tfidf_title_vectorizer.vocabulary_:
                labels.append(tfidf_title_features[doc_id, tfidf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)
        #print(labels)
    elif model == 'idf':
        labels = []
        for x in vec2.keys():
            if x in  idf_title_vectorizer.vocabulary_:
                labels.append(idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)
        #print(labels)

    plot_heatmap(keys, values, labels, url, text)

#converting the texts to vectors
def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words) 



def get_result(doc_id, content_a, content_b, url, model):
    text1 = content_a
    text2 = content_b
    
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    plot_heatmap_image(doc_id, vector1, vector2, url, text2, model)


# In[89]:


data1 = data.copy(deep=True)
temp = []
rows = data1[['medium_image_url','title']][0:100]
for indx, row in rows.iterrows():
    try:
        display(Image(url=row['medium_image_url'], embed=True))
    except(TypeError):
        temp.append('hello')
        
    


# In[90]:


print(temp)


# In[95]:


from sklearn.feature_extraction.text import CountVectorizer
title_vectorizer = CountVectorizer() #Takes the sentence and converts it into a vector with the frequency of the words.
title_features   = title_vectorizer.fit_transform(data['title'])
title_features.get_shape()


# In[96]:


# to extract the index of row of the product
asin_values = tuple(data['asin'])

value = asin_values.index('B01CLS8LMW')
print(value)


# In[98]:


BOW_asin=[]
index_BOW=[]

def bag_of_words_model(doc_id, num_results):
    pairwise_dist = pairwise_distances(title_features,title_features[doc_id]) 
    #Euclidian Distances between sample document  and all the other docments
    
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    df_indices = list(data.index[indices])
    
    if(len(BOW_asin)>0):
        BOW_asin.clear()
    if(len(index_BOW)>0):
        index_BOW.clear()
    
    for i in range(0,len(indices)):
        BOW_asin.append(data['asin'].loc[df_indices[i]])
        index_BOW.append(asin_values.index(data['asin'].loc[df_indices[i]]))
        get_result(indices[i],data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'bag_of_words')
        print('ASIN :',data['asin'].loc[df_indices[i]])
        print ('Brand:', data['brand'].loc[df_indices[i]])
        print ('Title:', data['title'].loc[df_indices[i]])
        print ('Euclidean similarity with the query image :', pdists[i])
        print('='*60)
#12566
bag_of_words_model(3971, 50)


# In[66]:


tfidf_title_vectorizer = TfidfVectorizer(min_df = 0) 
# statistical measure used to evaluate how important a word is to a document in a collection or corpus.
tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.vocabulary_)
print(X)


# In[99]:


TFIDF_asin = []
index_TFIDF = []

# making the tfidf model
def tfidf_model(doc_id, num_results):
    
    pairwise_dist = pairwise_distances(tfidf_title_features,tfidf_title_features[doc_id])

    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    df_indices = list(data.index[indices])
    
    if(len(TFIDF_asin)>0):
        TFIDF_asin.clear()
    if(len(index_TFIDF)>0):
        index_TFIDF.clear()

    for i in range(0,len(indices)):
        get_result(indices[i], data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'tfidf')
        print('ASIN :',data['asin'].loc[df_indices[i]])
        print('BRAND :',data['brand'].loc[df_indices[i]])
        print ('Eucliden distance from the given image :', pdists[i])
        print('='*125)
        TFIDF_asin.append(data['asin'].loc[df_indices[i]])
        index_TFIDF.append(asin_values.index(data['asin'].loc[df_indices[i]]))
        
tfidf_model(3971, 50)


# In[13]:


print(TFIDF_asin,index_TFIDF) #test


# In[100]:


idf_title_vectorizer = CountVectorizer()
idf_title_features = idf_title_vectorizer.fit_transform(data['title'])


# In[101]:


def n_containing(word):
    return sum(1 for blob in data['title'] if word in blob.split())

def idf(word):
    return math.log(data.shape[0] / (n_containing(word)))


# In[16]:


# takes time to execute
idf_title_features  = idf_title_features.astype(np.float)

for i in idf_title_vectorizer.vocabulary_.keys():
    idf_val = idf(i)
    
    for j in idf_title_features[:, idf_title_vectorizer.vocabulary_[i]].nonzero()[0]:
        
        idf_title_features[j,idf_title_vectorizer.vocabulary_[i]] = idf_val
        


# In[102]:


IDF_asin=[]
index_IDF=[]

def idf_model(doc_id, num_results):
    
    pairwise_dist = pairwise_distances(idf_title_features,idf_title_features[doc_id])
    
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]

    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    df_indices = list(data.index[indices])
    
    print(indices)
    
    if(len(IDF_asin)>0):
        IDF_asin.clear()
    if(len(index_IDF)>0):
        index_IDF.clear()
    
    for i in range(0,len(indices)):
        get_result(indices[i],data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'idf')
        print('ASIN :',data['asin'].loc[df_indices[i]])
        print('Brand :',data['brand'].loc[df_indices[i]])
        print ('euclidean distance from the given image :', pdists[i])
        print('='*125)
        IDF_asin.append(data['asin'].loc[df_indices[i]])
        index_IDF.append(asin_values.index(data['asin'].loc[df_indices[i]]))

value = asin_values.index('B00JXQASS6')

idf_model(3971,50)


# # Image based analysis

# In[103]:


# necessary imports
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import requests
import PIL
import PIL.Image
import pandas as pd
import pickle


# In[ ]:


# dimensions of our images.
# don't execute this
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'images2/'
nb_train_samples = 16042
epochs = 50
batch_size = 1


def save_bottlebeck_features():
    
    #Function to compute VGG-16 CNN for image feature extraction.
    
    asins = []
    datagen = ImageDataGenerator(rescale=1. / 255)
    
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    for i in generator.filenames:
        asins.append(i[2:-5])

    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
    bottleneck_features_train = bottleneck_features_train.reshape((16042,25088))
    
    np.save(open('16k_data_cnn_features.npy', 'wb'), bottleneck_features_train)
    np.save(open('16k_data_cnn_feature_asins.npy', 'wb'), np.array(asins))
    

save_bottlebeck_features()


# In[105]:


image_asin=[]
index_image=[]

#load the features and corresponding ASINS info.
bottleneck_features_train = np.load('F:\\final year project\\phase 2\\work\\16k_data_cnn_features.npy')
asins = np.load('F:\\final year project\\phase 2\\work\\16k_data_cnn_feature_asins.npy')
asins = list(asins)

# load the original 16K dataset
data = pd.read_pickle('F:\\final year project\\phase 2\\work\\pickels\\16k_apperal_data_preprocessed')
df_asins = list(data['asin'])


from IPython.display import display, Image

#get similar products using CNN features (VGG-16)
def get_similar_products_cnn(doc_id, num_results):
    doc_id = asins.index(df_asins[doc_id])
    pairwise_dist = pairwise_distances(bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1,-1))

    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    if(len(image_asin)>0):
        image_asin.clear()
    if(len(index_image)>0):
        index_image.clear()
    
    for i in range(len(indices)):
        rows = data[['medium_image_url','title']].loc[data['asin']==asins[indices[i]]]
        for indx, row in rows.iterrows():
            try:
                display(Image(url=row['medium_image_url'], embed=True))
            except(TypeError):
                pass
            print('Product Title: ', row['title'])
            print('Euclidean Distance from input image:', pdists[i])
            print('Amazon Url: www.amzon.com/dp/'+ asins[indices[i]])
            image_asin.append(asins[indices[i]])
            index_image.append(asin_values.index(asins[indices[i]]))
            
# 7871
get_similar_products_cnn(3971, 50)


# In[25]:


print(image_asin)
print(IDF_asin)
final_val_asin = [x for x in image_asin if x in IDF_asin]
print(final_val_asin)


# In[77]:


print(image_asin)
print(TFIDF_asin)
final_val_asin = [x for x in image_asin if x in TFIDF_asin]
print(final_val_asin2)


# In[30]:


print(image_asin)
print(BOW_asin)
final_val_asin = [x for x in image_asin if x in BOW_asin]
print(final_val_asin3)


# In[106]:


#Image and IDF combination
from IPython.display import display, Image
final_val_asin = []
final_val_index = []

final_val_asin = [x for x in image_asin if x in IDF_asin]
final_val_index = [x for x in index_image if x in index_IDF]

#print(final_val_asin,final_val_index)

for i in range(len(final_val_asin)):
    rows1 = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name',                 'title', 'formatted_price']].loc[data['asin']==final_val_asin[i]]
    for indx, row in rows1.iterrows():
        try:
            display(Image(url=row['medium_image_url'], embed=True))
        except(TypeError):
            pass
        print('Product Title: ', row['title'])
        print('Amazon Url: www.amzon.com/dp/'+ final_val_asin[i])


# In[107]:


#Image and TF-IDF combination
from IPython.display import display, Image
final_val_asin2 = []
final_val_index2 = []

final_val_asin2 = [x for x in image_asin if x in TFIDF_asin]
final_val_index2 = [x for x in index_image if x in index_TFIDF]

print(final_val_asin2,final_val_index2)

for i in range(len(final_val_asin2)):
    rows1 = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name',                 'title', 'formatted_price']].loc[data['asin']==final_val_asin2[i]]
    for indx, row in rows1.iterrows():
        try:
            display(Image(url=row['medium_image_url'], embed=True))
        except(TypeError):
            pass
        print('Product Title: ', row['title'])
        print('Amazon Url: www.amzon.com/dp/'+ final_val_asin2[i])


# In[108]:


#Image and BOW combination
from IPython.display import display, Image
final_val_asin3 = []
final_val_index3 = []
'''
j = 0
for i in image_asin:
    final_val_asin3.append(i)
    final_val_index3.append(index_image[j])
    j+=1

j = 0
for i in BOW_asin:
    if not i in final_val_asin3:
        final_val_asin3.append(i)
        final_val_index3.append(index_BOW[j])
        j+=1


final_val_asin3 = final_val_asin3[0:len(image_asin)]
final_val_index3 = final_val_index3[0:len(image_asin)]

random.shuffle(final_val_asin3)
'''

random.shuffle(image_asin)
random.shuffle(BOW_asin)

final_val_asin3 = [x for x in image_asin if x in BOW_asin]
final_val_index3 = [x for x in index_image if x in index_BOW]

#print(final_val_asin3,final_val_index3)

for i in range(len(final_val_asin3)):
    rows1 = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name',                 'title', 'formatted_price']].loc[data['asin']==final_val_asin3[i]]
    for indx, row in rows1.iterrows():
        try:
            display(Image(url=row['medium_image_url'], embed=True))
        except(TypeError):
            pass
        print('Product Title: ', row['title'])
        print('Amazon Url: www.amzon.com/dp/'+ final_val_asin3[i])

