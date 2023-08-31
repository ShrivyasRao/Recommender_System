# necessary imports
# importing all the dependencies

from flask import render_template
from app import app
from flask import Flask, redirect
import pickle
from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from scipy import sparse
from IPython.display import display, Image, SVG, Math, YouTubeVideo
import PIL
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
import sys
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
# image imports


app = Flask(__name__)

image_url = []

BOW_asin = []
index_BOW = []
TFIDF_asin = []
index_TFIDF = []
IDF_asin = []
index_IDF = []

image_asin = []
index_image = []


data = pd.read_pickle(
    'C:\\Users\\Shrivyas\\Desktop\project\\all_proj_fold\\final_proj_data\\pickels\\16k_apperal_data_preprocessed')

#Extract asins
asin_values = list(data['asin'])

# Takes the sentence and converts it into a vector with the frequency of the words.
title_vectorizer = CountVectorizer()
title_features = title_vectorizer.fit_transform(data['title'])
idf_title_vectorizer = CountVectorizer()
idf_title_features = idf_title_vectorizer.fit_transform(data['title'])
# statistical measure used to evaluate how important a word is to a document in a collection or corpus.
tfidf_title_vectorizer = TfidfVectorizer(min_df=0)
tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])

# BOW model


def bag_of_words_model(doc_id, num_results):
    pairwise_dist = pairwise_distances(title_features, title_features[doc_id])
    # Euclidian Distances between sample document  and all the other docments
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists = np.sort(pairwise_dist.flatten())[0:num_results]
    df_indices = list(data.index[indices])
    if(len(BOW_asin) > 0):
        BOW_asin.clear()

    if(len(index_BOW) > 0):
        index_BOW.clear()

    for i in range(0, len(indices)):
        BOW_asin.append(data['asin'].loc[df_indices[i]])
        index_BOW.append(asin_values.index(data['asin'].loc[df_indices[i]]))
        image_url.append(str(data['medium_image_url'].loc[df_indices[i]]))

# TFIDF model
# making the tfidf model


def tfidf_model(doc_id, num_results):

    pairwise_dist = pairwise_distances(
        tfidf_title_features, tfidf_title_features[doc_id])

    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists = np.sort(pairwise_dist.flatten())[0:num_results]
    df_indices = list(data.index[indices])

    if(len(TFIDF_asin) > 0):
        TFIDF_asin.clear()
    if(len(index_TFIDF) > 0):
        index_TFIDF.clear()

    for i in range(0, len(indices)):
        TFIDF_asin.append(data['asin'].loc[df_indices[i]])
        index_TFIDF.append(asin_values.index(data['asin'].loc[df_indices[i]]))


# IDF model
idf_title_vectorizer = CountVectorizer()
idf_title_features = idf_title_vectorizer.fit_transform(data['title'])


def n_containing(word):
    return sum(1 for blob in data['title'] if word in blob.split())


def idf(word):
    return math.log(data.shape[0] / (n_containing(word)))

# idf method


idf_title_features = sparse.load_npz(
    "C:\\Users\\Shrivyas\\Desktop\project\\all_proj_fold\\final_proj_data\\pickels\\yourmatrix.npz")


def idf_model(doc_id, num_results):

    pairwise_dist = pairwise_distances(
        idf_title_features, idf_title_features[doc_id])

    indices = np.argsort(pairwise_dist.flatten())[0:num_results]

    pdists = np.sort(pairwise_dist.flatten())[0:num_results]
    df_indices = list(data.index[indices])

    if(len(IDF_asin) > 0):
        IDF_asin.clear()
    if(len(index_IDF) > 0):
        index_IDF.clear()
    for i in range(0, len(indices)):
            # get_result(indices[i],data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'idf')
        IDF_asin.append(data['asin'].loc[df_indices[i]])
        index_IDF.append(asin_values.index(
        data['asin'].loc[df_indices[i]]))

# value = asin_values.index('B00JXQASS6')


# IMAGE PART
# load the features and corresponding ASINS info.
bottleneck_features_train = np.load(
    'C:\\Users\\Shrivyas\\Desktop\project\\all_proj_fold\\final_proj_data\\16k_data_cnn_features.npy')
asins = np.load(
    'C:\\Users\\Shrivyas\\Desktop\project\\all_proj_fold\\final_proj_data\\16k_data_cnn_feature_asins.npy')
asins = list(asins)

# load the original 16K dataset
data = pd.read_pickle(
    'C:\\Users\\Shrivyas\\Desktop\project\\all_proj_fold\\final_proj_data\\pickels\\16k_apperal_data_preprocessed')
df_asins = list(data['asin'])

# get similar products using CNN features (VGG-16)


def get_similar_products_cnn(doc_id, num_results):
    doc_id = asins.index(df_asins[doc_id])
    pairwise_dist = pairwise_distances(
        bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1, -1))

    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists = np.sort(pairwise_dist.flatten())[0:num_results]

    if(len(image_asin) > 0):
        image_asin.clear()
    if(len(index_image) > 0):
        index_image.clear()

    for i in range(len(indices)):
        rows = data[['medium_image_url', 'title']
                    ].loc[data['asin'] == asins[indices[i]]]
        for indx, row in rows.iterrows():
            try:
                pass
                # display(Image(url=row['medium_image_url'], embed=True))
            except(TypeError):
                pass
            image_asin.append(asins[indices[i]])
            index_image.append(asin_values.index(asins[indices[i]]))



# Image and BOW combination

final_val_asin = []
final_val_index = []

def image_bow():
    final_val_asin = [x for x in image_asin if x in BOW_asin]
    final_val_index = [x for x in index_image if x in index_BOW]
    
    final_img_links = []

    for val in BOW_asin:
        if val not in final_val_asin and len(final_val_asin) < len(BOW_asin):
            final_val_asin.append(val)
    for val in image_asin:
        if val not in final_val_asin and len(final_val_asin) < len(image_asin):
            final_val_asin.append(val)
    
    for i in range(len(final_val_asin)):
        rows1 = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name','title', 'formatted_price']].loc[data['asin'] == final_val_asin[i]]
        temp = rows1['medium_image_url']
        for j, row in temp.iteritems():
            final_img_links.append(row)
    return final_img_links

# Image and TF-IDF combination
final_val_asin2 = []
final_val_index2 = []

def image_tfidf():
    final_val_asin2 = [x for x in image_asin if x in TFIDF_asin]
    final_val_index2 = [x for x in index_image if x in index_TFIDF]

    final_img_links = []

    for val in TFIDF_asin:
        if val not in final_val_asin2 and len(final_val_asin2)<len(TFIDF_asin):
            final_val_asin2.append(val)
    for val in image_asin:
        if val not in final_val_asin2 and len(final_val_asin2)<len(image_asin):
            final_val_asin2.append(val)
    
    for i in range(len(final_val_asin2)):
        rows1 = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name','title', 'formatted_price']].loc[data['asin'] == final_val_asin2[i]]
       
        temp = rows1['medium_image_url']
    
        for j, row in temp.iteritems():
            final_img_links.append(row)

    return final_img_links

# Image and IDF combination

final_val_asin3 = []
final_val_index3 = []

def image_idf():
    
    final_val_asin3 = [x for x in image_asin if x in IDF_asin]

    final_val_index3 = [x for x in index_image if x in index_IDF]
    final_img_links = []
    for val in IDF_asin:
        if val not in final_val_asin3 and len(final_val_asin3)<len(IDF_asin):
            final_val_asin3.append(val)
    for val in image_asin:
        if val not in final_val_asin3 and len(final_val_asin3)<len(image_asin):
            final_val_asin3.append(val)
        
    for i in range(len(final_val_asin3)):
        rows1 = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name','title', 'formatted_price']].loc[data['asin'] == final_val_asin3[i]]
        
        
        temp = rows1['medium_image_url']
        
        for j, row in temp.iteritems():
            final_img_links.append(row)


    return final_img_links


@app.route('/')
def index():
    titles = []
    asin_val = []
    temp = data['asin'][12566:12607]
    
    for i, val in temp.iteritems():
        if val != 'B072QV5BNP':
            asin_val.append(str(val))
    
    urls = []
    err = []
    
    temp = data['title'][12566:12607]
    
    for i, j in temp.iteritems():
        if j != 'big buddhaedith cross body bag blush ':
            titles.append(str(j))
    temp = data['medium_image_url'][12566:12607]

    for i, j in temp.iteritems():
        if j != 'https://images-na.ssl-images-amazon.com/images/I/51GPn%2BcKhjL._SL160_.jpg':
            try:
                response = requests.get(j)
                img = PIL.Image.open(BytesIO(response.content))
                urls.append(j)
            except IOError:
                err.append(j)

    return render_template("index.html", len=len(titles), asins=asin_val, titles=titles, urls=urls, err=err)


@app.route('/<asinVal>')
def recommend(asinVal):
    value = 12566
    try:
        value = asin_values.index(asinVal)
    except ValueError:
        pass
    bag_of_words_model(value, 20)
    tfidf_model(value,20)
    idf_model(value,20)
    get_similar_products_cnn(value,20)

    #combinations B00JXQASS6
    final_val_asin = image_bow()
    final_val_asin2 =image_tfidf()
    final_val_asin3 = image_idf()

    return render_template("recommend.html", bow=final_val_asin, tfidf=final_val_asin2, idf=final_val_asin3)


if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=8000)
