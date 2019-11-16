# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:31:01 2019

@author: Tripti Santani
"""

import luigi
import pandas
import os
from Feature import feature_builder
from regressor import train_model_ridge,train_random_forest,train_model_with_grid_search
import pickle
from urllib2 import Request,urlopen,URLError

class TrainDataIngestion(luigi.Task):

    def run(self):
        #Importing the data
        listings=pd.read_csv('C:/Users/Tripti Santani/Desktop/Final Project/listings.csv',error_bad_lines=False)


    def output(self):
        return luigi.LocalTarget("C:/Users/Tripti Santani/Desktop/Final Project/listings.csv")


#class StoreDataIngestion(luigi.Task):

#    def run(self):
#        df_stores_data = pandas.read_csv(os.path.join(os.getcwd(), "data", "store.csv"))
#        df_stores_data.to_csv(self.output().path, index=False)

#    def output(self):
#        return luigi.LocalTarget("C:/Users/Tripti Santani/Desktop/Final Project/result.csv")


class AggregateTrainData(luigi.Task):

    def requires(self):
        yield TrainDataIngestion()
       # yield StoreDataIngestion()

    def run(self):
        df_train = pandas.read_csv(TrainDataIngestion().output().path)
        #df_stores = pandas.read_csv(StoreDataIngestion().output().path)
        #df_aggregated = pandas.merge(df_train, df_stores, on='Store')
        df_train.to_csv(self.output().path, index=False)

    def output(self):
        return luigi.LocalTarget("C:/Users/Tripti Santani/Desktop/Final Project/df_train_aggregated.csv")


class DataPreProcessing(luigi.Task):

    def requires(self):
        return AggregateTrainData()

    def run(self):
        fb = FeatureBuilder(pandas.read_csv(AggregateTrainData().output().path))
        dataframe = fb.featurize()
        print ("In Data Pre Processing")
        print (dataframe.columns)
        dataframe.to_csv(self.output().path, index=False)

    def output(self):
        return luigi.LocalTarget("C:/Users/Tripti Santani/Desktop/Final Project/rossman_sales_train_clean.csv")


class Train(luigi.Task):

    def requires(self):
        return DataPreProcessing()

    def run(self):
        sales_model = train_random_forest(pandas.read_csv(DataPreProcessing().output().path))
        with open(self.output().path, 'wb') as f:
            pickle.dump(sales_model,f )
        request = Request('http://127.0.0.1:5000/loadModels/')
        try:
            print ("Reloading models")
            response = urlopen(request)
        except (URLError, e):
            print("No Roseman Sales Prediction API", e)



    def output(self):
        return luigi.LocalTarget("/tmp/rossman_sales_model.pkl")


if __name__ == '__main__':
    luigi.run()