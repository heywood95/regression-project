# Regression-Project
 
# Project Description
 
The property values for Zillow will be investigated using a variety of target variables to predict what affects property value using regression techniques, and make predictions on property values. I have decided to look into four different areas that may affect property value.
 
# Project Goal
 
* Find drivers for customer churn at Telco. Why are customers churning?
* Construct a ML classification model that accurately predicts customer churn.
* Deliver a report that a non-data scientist can read through and understand what steps were taken, why and what was the outcome?

 
# Initial Thoughts
 
My initial hypothesis is that drivers of churn will depend on the type of service used by a customer and the amount of the montly payments.
 
# The Plan
 
* Aquire data from telco_churn database
 
* Prepare data
   * Cleaned up data
       * Dropped unnecessary columns  
       * Renamed confusing columns  
       * Dropped duplicate columns  
       * Replaced null values in the hottub_spa, pools, garages, and stories columns  
       * Eliminated outliers in the square_footage, bedrooms, bathrooms, and property_value columns   
       * Dropped null values  
       * split the data  
 
* Explore data in search of drivers of upsets
   * Answer the following initial questions
       * Does the square footage of a property increase, decrease, or is irrelevant in 
         determining the property value.
       * What number of bedrooms is optimal for the property value of a house
       * Does the number of bathrooms increase the property value?
       * Does the zipcode effect the lot size and is this a determining factor in the 
         property value.
      
* Develop a Model to predict the property value of a house
   * Use drivers identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best model based on highest validate and difference accuracy
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|'airconditioningtypeid'|	 Type of cooling system present in the home (if any)|
|'architecturalstyletypeid'|	Architectural style of the home (i.e. ranch, colonial, split-level, etc…)|
|'basementsqft'| Finished living area below or partially below ground level|
|'bathroomcnt'|	Number of bathrooms in home including fractional bathrooms|
|'bedroomcnt'|	Number of bedrooms in home| 
|'buildingqualitytypeid'|	|Overall assessment of condition of the building from best (lowest) to worst (highest)|
|'buildingclasstypeid'|	The building framing type (steel frame, wood frame, concrete/brick)| 
|'calculatedbathnbr'|	Number of bathrooms in home including fractional bathroom|
|'decktypeid'|	Type of deck (if any) present on parcel|
|'threequarterbathnbr'|	Number of 3/4 bathrooms in house (shower + sink + toilet)|
|'finishedfloor1squarefeet'|	Size of the finished living area on the first (entry) floor of the home|
|'calculatedfinishedsquarefeet'|	Calculated total finished living area of the home| 
|'finishedsquarefeet6'|	Base unfinished and finished area|
|'finishedsquarefeet12'|	Finished living area|
|'finishedsquarefeet13'|	Perimeter  living area|
|'finishedsquarefeet15'|	Total area|
|'finishedsquarefeet50'|	Size of the finished living area on the first (entry) floor of the home|
|'fips'|	Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details|
|'fireplacecnt'|	Number of fireplaces in a home (if any)|
|'fireplaceflag'|	Is a fireplace present in this home| 
|'fullbathcnt'|	Number of full bathrooms (sink, shower + bathtub, and toilet) present in home|
|'garagecarcnt'|	Total number of garages on the lot including an attached garage|
|'garagetotalsqft'|	Total number of square feet of all garages on lot including an attached garage|
|'hashottuborspa'| Does the home have a hot tub or spa|
|'heatingorsystemtypeid'|	Type of home heating system|
|'latitude'|	Latitude of the middle of the parcel multiplied by 10e6|
|'longitude'|	Longitude of the middle of the parcel multiplied by 10e6|
|'lotsizesquarefeet'|	Area of the lot in square feet|
|'numberofstories'|	Number of stories or levels the home has|
|'parcelid'|	Unique identifier for parcels (lots)| 
|'poolcnt'|	Number of pools on the lot (if any)|
|'poolsizesum'|	Total square footage of all pools on property|
|'pooltypeid10'|	Spa or Hot Tub|
|'pooltypeid2'|	Pool with Spa/Hot Tub|
|'pooltypeid7'|	Pool without hot tub|
|'propertycountylandusecode'|	County land use code i.e. it's zoning at the county level|
|'propertylandusetypeid'|	Type of land use the property is zoned for|
|'propertyzoningdesc'|	Description of the allowed land uses (zoning) for that property|
|'rawcensustractandblock'|	Census tract and block ID combined - also contains blockgroup assignment by extension|
|'censustractandblock'|	Census tract and block ID combined - also contains blockgroup assignment by extension|
|'regionidcounty'|	County in which the property is located|
|'regionidcity'|	City in which the property is located (if any)|
|'regionidzip'|	Zip code in which the property is located|
|'regionidneighborhood'|	Neighborhood in which the property is located|
|'roomcnt'|	Total number of rooms in the principal residence|
|'storytypeid'|	Type of floors in a multi-story house (i.e. basement and main level, split-level, attic, etc.).  See tab for details.|
|'typeconstructiontypeid'|	What type of construction material was used to construct the home|
|'unitcnt'|	Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...)|
|'yardbuildingsqft17'|	Patio in  yard|
|'yardbuildingsqft26'|	Storage shed/building in yard|
|'yearbuilt'|	The Year the principal residence was built| 
|'taxvaluedollarcnt'|	The total tax assessed value of the parcel|
|'structuretaxvaluedollarcnt'|	The assessed value of the built structure on the parcel|
|'landtaxvaluedollarcnt'|	The assessed value of the land area of the parcel|
|'taxamount'|	The total property tax assessed for that assessment year|
|'assessmentyear'|	The year of the property tax assessment| 
|'taxdelinquencyflag'|	Property taxes for this parcel are past due as of 2015|
|'taxdelinquencyyear'|	Year for which the unpaid propert taxes were due| 
 
# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from Codeup DB Server/zillow
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* About 43% of customers with fiber optic churned.
* About 13% of customers without fiber optic churned.
* It is more likely that customers who have fiber optic churned versus customers who did not have fiber optic.
* Month to month contracts have the highest level of churn.
* One year contracts are the next highest, but still relatively low churn.
* Two year contracts have the lowest level of churn.
* While it is no surprise that month to month contracts produce the highest level of churn, the drastic drop from month to month to one       year and two year contracts should be looked at and possible incentives or discounts for the one year and two year contracts could reduce   the amount of month to month contracts and level of churn.
* Customers with online security have 17% churn rate.
* Customers with no internet service have an 11% churn rate.
* Customers withouth online security have a 40% churn rate.
* The major contributor for customer churn is more correlated with internet services that provide security such as tech support, device       protection, online backup, and online security.
* Other services for entertainment purposes such as streaming tv and movies had less of an impact on customer churn.
* As charges increase, churn increases until the rate reaches over 100 dollars.
* The largest proportion of customers spent between 20-30 dollars.
* It would appear that the more money that was spent by the customer, the more likely they are to churn.
* The large majority of those that churned spent 500 dollars or less on total charges.
* Those customers who spent about 80 dollars a month were the most likely to churn.
* Total charges produced insufficient data to support a correlation between total charges and churn.
 
# Recommendations
* To reduce churn decrease the amount of month to month contracts and increase the amount of one and two year contracts.
* Provide online security to all of the customers, this area produced the highest probability of customer churn.
