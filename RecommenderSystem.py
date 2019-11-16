# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 08:47:39 2019

@author: Tripti Santani
"""

# Importing the libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

listing = pd.read_csv('C:/Users/Tripti Santani/Desktop/Gitupload/finalproject/listings.csv')

def dataClean(listing):
    
    listing.price=listing.price.fillna(listing.price.mean)
    listing['price'] = pd.to_numeric(listing['price'].apply(lambda x: str(x).replace('$', '')),errors='coerce')
    
    listings = listing[['id', 'name', 'description']]
    
    listings['name'] = listings['name'].astype('str')
    listings['description'] = listings['description'].astype('str')
    
    listings['content'] = listings[['name', 'description']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
    
    listings['content'].fillna('Null', inplace = True)
    return listings

def tfIdf(listings):
  
    tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
    tfidf_matrix = tf.fit_transform(listings['content'])
    return tfidf_matrix
    
def cosineSimilarities(tfidf_matrix):
    
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    #Iterate through each item's similar items and store the 100 most-similar!
    results = dict()
    for idx, row in listings.iterrows():
        
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], listings['id'][i]) for i in similar_indices]
        results[row['id']] = similar_items[1:]
        
    return results
    
def item(id):
    discount=0
    name   = listings.loc[listings['id'] == id]['content'].tolist()[0].split(' // ')[0]
    desc   = ' \nDescription: ' + listings.loc[listings['id'] == id]['content'].tolist()[0].split(' // ')[1][0:165] + '...'
    for i in range(len(listing)):
      if(listing.id[i]==id):
        if (listing.price[i]>500):
          discount = listing.price[i]
          dis=str(discount)+'discount of 20%'
        elif(listing.price[i]>=300 and listing.price[i]<500):
          discount=listing.price[i]
          dis = str(discount) + 'discount of 15%'
        elif (listing.price[i]>=150 and listing.price[i]<300):
          discount=listing.price[i]
          dis = str(discount) + 'discount of 10%'
        elif(listing.price[i]<150):
          discount=listing.price[i]
          dis=str(discount)+ 'discount of 5%'
     
    prediction = str(name  + desc + dis)
    return prediction

def location_recommender(id, num):
    listings = dataClean(listing)
    tfidf_matrix = tfIdf(listings)
    results=cosineSimilarities(tfidf_matrix)
    recs = results[id][:num]
    for rec in recs:
        return(item(rec[1]))
    
recommend(id = 6990882, num = 5)

