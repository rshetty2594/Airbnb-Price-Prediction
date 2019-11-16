# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:47:29 2019

@author: Tripti Santani
"""

import numpy as np
import pandas as pd


class feature_builder():
    
    def feature(dataframe=df):

        listings['monthly_price'] = pd.to_numeric(listings['monthly_price'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')
        listings['price'] = pd.to_numeric(listings['price'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')
        listings['weekly_price'] = pd.to_numeric(listings['weekly_price'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')
        listings['security_deposit'] = pd.to_numeric(listings['security_deposit'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')
        listings['cleaning_fee'] = pd.to_numeric(listings['cleaning_fee'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')
        listings['extra_people'] = pd.to_numeric(listings['extra_people'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')
        listings['host_response_rate'] = pd.to_numeric(listings['host_response_rate'].apply(lambda x: str(x).replace('%', '')),errors='coerce')
        
        df = listings[["host_response_rate", "host_acceptance_rate", "host_is_superhost","security_deposit","cleaning_fee","extra_people",
                       "host_listings_count", "zipcode", "property_type","room_type", "accommodates", "bathrooms", "bedrooms",
                       "beds", "price", "number_of_reviews", "review_scores_rating", "cancellation_policy", 
                       "reviews_per_month"]]
        # Handling the missing values:
        df.price=df.price.fillna(df.price.mean)
        df.security_deposit=df.security_deposit.fillna(0)
        df.reviews_per_month=df.reviews_per_month.fillna(0)
        df.cleaning_fee=df.cleaning_fee.fillna(0)
        
        # drop NaN rows
        df2=df.dropna(axis=0)
        
        pd.options.mode.chained_assignment = None  # default='warn'
        df2['host_response_rate'] = df2['host_response_rate'].astype(str)
        df2['host_acceptance_rate'] = df2['host_acceptance_rate'].astype(str)
        df2['price'] = df2['price'].astype(str)
        
        # clean data
        pd.options.mode.chained_assignment = None  # default='warn'
        df2['host_acceptance_rate'] = df2['host_acceptance_rate'].str.replace("%", "").astype("float")
        df2['price'] = df2['price'].str.replace("[$, ]", "").astype("float")
        df2['host_response_rate'] = df2['host_response_rate'].str.replace("%", "").astype("float")
        
        df2['superhost']=np.where(df2['host_is_superhost']=='t',1,0)
        del df2['host_is_superhost']
        
        #Converting into numeric data
        df2['bedrooms'] = pd.to_numeric(df2['bedrooms'],errors='coerce')
        df2['accommodates'] = pd.to_numeric(df2['accommodates'],errors='coerce')
        df2['bathrooms'] = pd.to_numeric(df2['bathrooms'],errors='coerce')
        df2['number_of_reviews']=pd.to_numeric(df2['number_of_reviews'],errors='coerce')
        df2['beds']=pd.to_numeric(df2['beds'],errors='coerce')
        df2['number_of_reviews']=pd.to_numeric(df2['number_of_reviews'],errors='coerce')
        df2['host_listings_count']=pd.to_numeric(df2['host_listings_count'],errors='coerce')
        df2['review_scores_rating']=pd.to_numeric(df2['review_scores_rating'],errors='coerce')
        df2['zipcode']=pd.to_numeric(df2['zipcode'],errors='coerce')
        
        # select non-numeric variables and create dummies
        non_num_vars = df2.select_dtypes(include=['object']).columns
        dummy_vars = pd.get_dummies(df2[non_num_vars])
        
        # drop non-numeric variables from df2 and add the dummies
        df3=df2.drop(non_num_vars,axis=1)
        df3 = pd.merge(df3,dummy_vars, left_index=True, right_index=True)
        
        #Droping the missing values
        
        df3=df3.dropna()




