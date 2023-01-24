#!/usr/bin/env python
# coding: utf-8

# In[ ]:

### ACQUIRE ###

import pandas as pd
import numpy as np
import os
from env import host, username, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def get_connection(db, user=username, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = """
            SELECT bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, fireplacecnt, garagecarcnt,
            hashottuborspa, lotsizesquarefeet, poolcnt, regionidzip, yearbuilt, numberofstories,
            taxvaluedollarcnt, propertylandusedesc, transactiondate
            FROM properties_2017 
            JOIN propertylandusetype USING(propertylandusetypeid)
            JOIN predictions_2017 USING(parcelid)
            WHERE propertylandusedesc = 'Single Family Residential' AND transactiondate >= '2017-01-01'
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def acquire_zillow():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:

        #creates new csv if one does not already exist
        df = get_zillow_data()
        df.to_csv('zillow.csv')

    return df
