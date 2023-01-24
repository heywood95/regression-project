#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from env import host, username, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_zillow(df):
    '''Prepares acquired zillow data for exploration'''
    
    # drop column using .drop(columns=column_name)
    df = df.drop(columns=['propertylandusedesc', 'transactiondate'])
    
    
    # rename columns
    df = df.rename(columns={'calculatedfinishedsquarefeet': 'square_footage', 'taxvaluedollarcnt': 'property_value', 'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'yearbuilt': 'year_built', 'fireplacecnt': 'fire_place', 'garagecarcnt': 'garage', 'hashottuborspa': 'hottub_spa', 'lotsizesquarefeet': 'lot_size', 'poolcnt': 'pools', 'regionidzip': 'zip_code', 'numberofstories': 'stories'})
    
    # add a column name optimal square footage
    optimal_square_footage = ([[df['square_footage'] >= 1001] and [df['square_footage'] <= 2000]])
    
    df['optimal_sf'] = optimal_square_footage[0][0]
    
    # Convert binary categorical variables to numeric
    df['optimal_sf'] = df.optimal_sf.map({True: 1, False: 0})
    
    # drop duplicates    
    df.drop_duplicates(inplace=True)
    # In[2]:

    # replace NULL with a specific value, either 0 or 1
    df['fire_place'].fillna(0, inplace = True)
    df['garage'].fillna(0, inplace = True)
    df['hottub_spa'].fillna(0, inplace = True)
    df['pools'].fillna(0, inplace = True)
    df['stories'].fillna(1, inplace = True)
    
    # get rid of outliers
    for x in ['property_value', 'square_footage', 'bedrooms', 'bathrooms']:
        q75,q25 = np.percentile(df.loc[:, x],[75,25])
        intr_qr = q75-q25
 
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
 
        df.loc[df[x] < min,x] = np.nan
        df.loc[df[x] > max,x] = np.nan
    
        # drop the nulls
        df = df.dropna(axis=0)
    
    # split the data
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test


def split_data(df):
    '''
    This function takes in a dataframe and splits the data,
    returning three pandas dataframes, train, test, and validate
    '''
    #create train_validate and test datasets
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    #create train and validate datasets
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123)
  
    return train, validate, test 



#create a function to isolate the target variable
def X_y_split(df, target):
    '''
    This function takes in a dataframe and a target variable
    Then it returns the X_train, y_train, X_validate, y_validate, X_test, y_test
    and a print statement with the shape of the new dataframes
    '''  
    train, validate, test = split_data(df)

    X_train = train.drop(columns= [target])
    y_train = train[target]

    X_validate = validate.drop(columns= [target])
    y_validate = validate[target]

    X_test = test.drop(columns= [target])
    y_test = test[target]
        
    # Have function print datasets shape
    print(f'X_train -> {X_train.shape}')
    print(f'X_validate -> {X_validate.shape}')
    print(f'X_test -> {X_test.shape}')  
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test



def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedrooms', 'bathrooms', 'square_footage', 'year_built', 'fire_place', 'garage', 'hottub_spa', 'lot_size', 'pools', 'zip_code', 'stories', 'optimal_sf'], return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    from sklearn.preprocessing import MinMaxScaler
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),                              columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),                        columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),                                columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
    
    
def wrangle_zillow():
    '''
    This function uses the acquire and prepare functions
    and returns the split/cleaned dataframe
    '''
    train, validate, test = prep_zillow(acquire_zillow())
    
        
    return train, validate, test


def rfe(X, y, n):
    '''
    This function takes in the features, target variable 
    and number of top features desired and returns a dataframe with
    the features ranked
    '''
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import SelectKBest, f_regression, RFE
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=n)
    rfe.fit(X, y)
    ranks = rfe.ranking_
    columns = X.columns.tolist()
    feature_ranks = pd.DataFrame({'ranking': ranks, 'feature': columns})
    return feature_ranks.sort_values('ranking')

def visual_correlations(df):
    '''
    This function creates a heatmap of the features
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    count_var = ['property_value', 'bathrooms', 'square_footage', 'bedrooms', 'optimal_sf']

    train_corr = df[count_var].corr()
    
    plt.title('Strength of correlation with property value')
    
    return sns.heatmap(train_corr, annot=True, annot_kws={"size": 7}, linewidths=2, linecolor='yellow', yticklabels=14)


def eval_result(df):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    from scipy import stats
    alpha = 0.05
    r, p = stats.spearmanr(df.square_footage, df.property_value)
    
    null_hypothesis = "the square footage of a property is irrelevant in determining property value."
    alternative_hypothesis = "the square footage of a property will either increase or decrease property value."
    
    if p < alpha:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis, "p-value=", p, "r=", r)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    
def eval_result2(df):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    from scipy import stats
    alpha = 0.05
    r, p = stats.spearmanr(df.bedrooms, df.property_value)
    
    null_hypothesis = "the property value of a house is not dependent on the number of bedrooms.   "
    alternative_hypothesis = "the property value of a house is dependent on the number of bedrooms."
    
    if p < alpha:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis, "p-value =", p, "r =", r)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")

        
def eval_result3(df):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    from scipy import stats
    alpha = 0.05
    r, p = stats.spearmanr(df.bathrooms, df.property_value)
    
    null_hypothesis = "the number of bathrooms does not affect the property value.   "
    alternative_hypothesis = "the number of bathrooms affects the property value."
    
    if p < alpha:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis, "p-value =", p, "r =", r)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null") 

        
        
def eval_result4update(df):
    '''
    This function conducts a spearmanr statistical test
    and returns the results.
    '''
    from scipy import stats
    alpha = 0.05
    r, p = stats.spearmanr(df.optimal_sf, df.property_value)
    
    null_hypothesis = "there is not an optimal square footage that correlates with property value.    "
    alternative_hypothesis = "there is an optimal square footage that correlates with property value."
    
    if p < alpha:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis, "p-value =", p, "r =", r)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
        
        
        
def eval_result4(df):
    '''
    This function conducts a ttest statistical test
    and returns the results.
    '''
    from scipy import stats
    alpha = 0.05
    
    t, p = stats.ttest_ind(df.optimal_sf, df.property_value)
    
    null_hypothesis = "there is not an optimal square footage that correlates with property value.    "
    alternative_hypothesis = "there is an optimal square footage that correlates with property value."
    
    if p < alpha:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis, "p-value =", p, "t =", t)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
        
        
def baseline(yt, yv):
    '''
    This function takes in y_train and y_validate and prints the mean and meadian 
    baselines.
    '''
    import sklearn.preprocessing 
    from sklearn.metrics import mean_squared_error
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    yt = pd.DataFrame(yt)
    yv = pd.DataFrame(yv)

    # 1. Predict G3_pred_mean
    property_value_pred_mean = yt['property_value'].mean()
    yt['property_value_pred_mean'] = property_value_pred_mean
    yv['property_value_pred_mean'] = property_value_pred_mean

    # 2. compute G3_pred_median
    property_value_pred_median = yt['property_value'].median()
    yt['property_value_pred_median'] = property_value_pred_median
    yv['property_value_pred_median'] = property_value_pred_median

    # 3. RMSE of G3_pred_mean
    rmse_train = mean_squared_error(yt.property_value, yt.property_value_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(yv.property_value, yv.property_value_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    # 4. RMSE of G3_pred_median
    rmse_train = mean_squared_error(yt.property_value, yt.property_value_pred_median)**(1/2)
    rmse_validate = mean_squared_error(yv.property_value, yv.property_value_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    
    
    
def linear_reg_model(Xt, yt, yv, Xv):
    '''
    This function creates, fits, and predicts the RMSE for a
    LinearRegression model and outputs the results.
    '''
    
    import sklearn.preprocessing
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    yt = pd.DataFrame(yt)
    yv = pd.DataFrame(yv)
    
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(Xt, yt.property_value)

    # predict train
    yt['property_value_pred_lm'] = lm.predict(Xt)

    # evaluate: rmse
    rmse_train = mean_squared_error(yt.property_value, yt.property_value_pred_lm)**(1/2)

    # predict validate
    yv['property_value_pred_lm'] = lm.predict(Xv)

    # evaluate: rmse
    rmse_validate = mean_squared_error(yv.property_value, yv.property_value_pred_lm)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)

    
    
    
def lasso_lars_model(Xt, yt, yv, Xv):
    '''
    This function creates, fits, and predicts the RMSE for a
    Lasso-Lars model and outputs the results.
    '''
    
    import sklearn.preprocessing
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LassoLars
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    yt = pd.DataFrame(yt)
    yv = pd.DataFrame(yv)
    
    # create the model object
    lars = LassoLars(alpha=1.0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(Xt, yt.property_value)

    # predict train
    yt['property_value_pred_lars'] = lars.predict(Xt)

    # evaluate: rmse
    rmse_train = mean_squared_error(yt.property_value, yt.property_value_pred_lars)**(1/2)

    # predict validate
    yv['property_value_pred_lars'] = lars.predict(Xv)

    # evaluate: rmse
    rmse_validate = mean_squared_error(yv.property_value, yv.property_value_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)

    
    
    
def poly_model(Xt, yt, yv, Xv):
    '''
    This function creates, fits, and predicts the RMSE for a
    polynomial model and outputs the results.
    '''
    
    import sklearn.preprocessing
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    yt = pd.DataFrame(yt)
    yv = pd.DataFrame(yv)
    
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=1)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(Xt)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(Xv)
    X_test_degree2 = pf.transform(Xt)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, yt.property_value)

    # predict train
    yt['property_value_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(yt.property_value, yt.property_value_pred_lm2)**(1/2)

    # predict validate
    yv['property_value_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(yv.property_value, yv.property_value_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)


    
    
    
def poly_test_model(Xtest, ytest):
    '''
    This function creates, fits, and predicts the RMSE for a
    polynomial test model and outputs the results.
    '''
    
    import sklearn.preprocessing
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    # We need y_test to be dataframes to append the new columns with predicted values. 
    ytest = pd.DataFrame(ytest)
    
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=1)

    # fit and transform X_train_scaled
    X_test_degree2 = pf.fit_transform(Xtest)

    # transform X_test_scaled
    X_test_degree2 = pf.transform(Xtest)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our test data. We must specify the column in y_test, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_test_degree2, ytest.property_value)

    # predict test
    ytest['property_value_pred_lm2'] = lm2.predict(X_test_degree2)

    # evaluate: rmse
    rmse_test = mean_squared_error(ytest.property_value, ytest.property_value_pred_lm2)**(1/2)
                                                                                          
    print("RMSE for Polynomial model\nOut-of-Sample Performance: ", rmse_test)

    
    
def sq_ft_visual(df):
    '''
    This function creates a relplot of the square footage
    and property value with bedrooms as the hue.
    '''
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    rel=sns.relplot(x='square_footage', y='property_value', data=df, hue='bedrooms',
            palette=['b', 'r', 'g', 'purple']).set_axis_labels("Square Footage","Property Value")
    
    return rel.fig.suptitle('Square footage increases property value')



def bed_visual(df):
    '''
    This function creates a barplot of the bedrooms
    and property value.
    '''
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    ax=sns.barplot(x='bedrooms', y='property_value', data=df)
    
    return ax.set(xlabel='Bedrooms', ylabel='Property Value', title='The number of bedrooms increases property value')



def bath_visual(df):
    '''
    This function creates a barplot of the bathrooms
    and property value.
    '''
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    ax2=sns.barplot(x='bathrooms', y='property_value', data=df, palette=['lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'red', 'lightgrey', 'lightgrey'])
    
    return ax2.set(xlabel='Bathrooms', ylabel='Property Value', title='Three and a half bathrooms have the highest increase')



def opt_sf_visual(df):
    '''
    This function creates a barplot of the bathrooms
    and property value.
    '''
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    ax3=sns.barplot(x='optimal_sf', y='property_value', data=df, palette=['lightgrey', 'r'])
    
    return ax3.set(xlabel='Square Footage', ylabel='Property Value', title='Optimal square footage is from 1000-2000 sq. ft.', xticklabels=('other sq. ft.', '1000-2000 sq. ft'))
# In[ ]:




