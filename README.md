# Regression-Project
 
# Project Description
 
The property values for Zillow will be investigated using a variety of target variables to predict what affects property value using regression techniques, and make predictions on property values. I have decided to look into four different areas that may affect property value.
 
# Project Goal
 
* Construct an ML Regression model that predicts propery tax assessed values of Single Family Properties using features of the properties.
* Find the key drivers of property value for single family properties. 
* Deliver a report that explains what steps were taken, why and what the outcome was.
* Make recommendations on what works or doesn't work in predicting these homes' values.

 
# Initial Thoughts
 
My initial hypothesis is that the property value will increase with square footage and the number of bedrooms and bathrooms.
 
# The Plan
 
* Aquire data from zillow database
 
* Prepare data
   * Cleaned up data
       * Dropped unnecessary columns  
       * Renamed confusing columns  
       * Dropped duplicate columns  
       * Replaced null values in the hottub_spa, pools, garages, and stories columns  
       * Eliminated outliers in the square_footage, bedrooms, bathrooms, and property_value columns   
       * Dropped null values  
       * split the data  
 
* Explore data in search of drivers of property value
   * Answer the following initial questions
       * Does the square footage of a property increase, decrease, or is irrelevant in 
         determining the property value.
       * What number of bedrooms is optimal for the property value of a house?
       * Does the number of bathrooms increase the property value?
       * Is there an optimal square footage for property value?
      
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
* Bathrooms, bedrooms, and square footage are the top three features with the strongest correlation.
    * Square footage has the strongest correlation of all the features.
* There is a moderate corellation between square footage and property value. 
    * Further investigation into size of the property and its value is necessary in determing the optimal square footage.
* There is a slight correlation between the number of bedrooms and property value. It is difficult to determine the maximum number of bedrooms based on the chart.
* The optimal number of bathrooms to optimize property value appears to be 3.5. However more investigation is required to confirm this finding. Further analysis on optimizing the cost of building a property, location, and comparing that the total value of the propery is neccessary.
* The optimal square footage is between 1001-2000 square feet. However, there is a moderate negative correlation and the large number of homes within this range could be causing this correlation.
 
# Recommendations
* The square footage of a house appears to be optimal between 1000-2000 square feet.
* The number of bedrooms does not apeear to be a significant factor in determining the property value. However, targeted analysis on specific bedroom counts could produce different conclusions.
* The number of optimal bathrooms appears to be 3.5, although more analysis into the increase in value is necessarry to confirm this conclusion.
