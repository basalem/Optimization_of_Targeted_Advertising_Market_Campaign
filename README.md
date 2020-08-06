
# Optimization of Targeted Advertising Market Camping  
Author: Mohammed Ba Salem 

Contact: basaleemm@gmail.com

LinkedIn: [https://www.linkedin.com/in/mohammed-basalem/](https://www.linkedin.com/in/mohammed-basalem/)

## Business Objective 
The following project is part of an interview showcase study for **XX_Consulting_Company** aiming to help the marketing department to develop an optimized financial strategy for a client's advertising market campaign through the utilization of ML algorithms. 

The machine learning model designed to predict how much profit the marketing team can make by accurately classify which of their target customers are most likely click on the ads. As part of data science cycle, statistical tests, feature selection, feature importance and data visualization are performed to understand how a particular feature contributes to the end results and how can those features help market team to run their campaign in optimized manner.    

## Dataset-Background 
For the following problem, no much details are given from the company. Thus, I have to assume or define some income statements that will be resulted from ML model: 

1. Client will spend **550 CAD** for a potenial customer. 
2. **Overall Profit** from targeted customer who clicks on ad is **150 CAD**. 
3. **Loss** from targeted customer who does not click on ad is **600 CAD**. 
4. **Profit** from un-targeted customer who clicks on ad is **700 CAD**.
5. Product is **un-known**.


The dataset contains website visitors information and their view of ads. It has 1018 rows and 10 columns as described below: 

- Daily Time Spent on Site: Time spent by user in the site in Minutes.
- Age.                         
- Area Income: Average income of the geographical area of user.                 
- Daily Internet Usage: Average time spent by user on the internet
- Ad Topic Line: Headline of the advertisement    
- City                        
- Male: gender, 1 for Male and 0 for Female.                        
- Country                     
- Timestamp: Data and time at which consumer clicked on the Ad or close window (not clicked)                  
- Clicked on Ad: Target variable, 1 for clicked(positive), 0 for not clicked(Negative).   


![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/Class_Distribution.PNG) 

## Libraries 

**Python Version**: 3.7.6

**Jupyter Notebook**: 6.0.3

**Packages**:  pandas, numpy, datetime, calendar, ppscore, scipy, sklearn, xgboost, matplotlib, seaborn.
 
## Methodology 

### Clean and Explore Data 
It is always important to ensure that we have a good quality and representative data to give it the meaning it deserves through EDA, feature engineering and ML modeling. Some pre-processing is needed to perform:

1. Explore data features and observations. 
2. Business logic validation in data. 
3. Check missing values and duplicates. 

### Exploratory Analysis and Visualization 
Features associations with  target classes have been visualization to obtain some information about data and to discover meaningful insights about customers behaviors. 
#### All Class Distribution 
Consumers who spend less time on site, less time on internet, low income class (less than 50k) and aged 25-50 are more likely to click on the ads. 

**Note:** Further details are presented in the Jupyter Notebook above. 
		
![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/All_Class_Distribution.PNG)

#### Clicked Class Distribution 
- This group has a low range of daily time spend on site where majority of consumers' time fall in range of **40-60 minutes** with **positively skewed** distribution. 
- The age distribution is normal withe mean and median centered at **40** years old.  
- The average area income has a normal distribution shape with mean centered at **50k**. 
- The distribution of daily internet usage is positively skewed where majoity of consumers spent time between **100-175** minutes.


![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/Clicked_Class_Distribution.PNG)	

#### Not Clicked Class Distribution 
- This group has a high range of daily time spend on site where majority of consumers' time fall in range of **70-90 minutes** with **negatively skewed** distribution. 
- The age distribution is almost approaching a normal distribution with mean around at **31.6**, having low positive skewness. 
- The average area income is almost approaching a normal distribution with mean around **60K**, having low negative skewness. 
- The distribution of daily internet usage is negatively skewed where majoity of consumers spent time between **200-260** minutes. 

![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/Not_Clicked_Class_Distribution.PNG)


![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/DateTime_Analysis.PNG)



![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/Date_Time_Analysis_Clicked_Class.PNG)


![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/Date_Time_Analysis_Not_Clicked_Class.PNG)

### Feature Selection 
It is always said gold in gold out, that is feeding your ML model with clean, good quality and highly representative data will give in return good results! To perform feature selection, we have to do first some feature transformation as an initial step, then feature selection. Statistical testing and machine learning approaches are used here. 

![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/Feature_Selection.png)


![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/Heat_Map_Selection.png)       

## Model Selection 
 4 different classification algorithms including a baseline mode are trained to predict who will click and who will not click on the ads. Algorithms used are *Naive Bayes(Baseline)*, *Logistic Regression* ,*Random Forest Classifier* and *XgBoost* . As the data is highly balance, it is necessary to select multiple metrics that can return a meaningful accuracy and compare results. AUPRC, AUC, F1-Score, Accuracy and F-Beta are select to determine best classifier. 

**Baseline and Logistic Regression Models**

![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/Images/BS_LR.png)

**Random Forest and XgBoost Models**

![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/RF_XG.png)


From above metrics, it is clearly seen that: 

- All classifiers do not suffer from overfitting or underfitting, that is there is a balance between bias and variance! 
- Most of the 3 classifiers overperform the Baseline on both training and testing sets. 
- From classification perspective, we can generate a lot of money from consumers who are not targeted by us, but they clicked on the add, **FP class,700CAD**. This can be explained as company is more likly popular or well know. 


## Results 
Before jumping to final model results and selection, I just wanted to re-define some parameters stated above in the business objective and related them to outcome from ML classifiers. This will give us an overall understanding of final results. 

**Business & Profit** 

As the business objective aims to predict how much profit the marketing team can make by accurately classify which of their targeted customers are more likely to click on the ads. I have defined some financial statements, and to present model performance from business perspective, I defined the following indicators: 

- **Positive class** is 1 = clicked on ad, **Negative class** is 0 = not clicked on ad. 


- *False Positive* (**FP**): It means actual class is Clicked **(consumer clicked on Ads)** and incorrectly classified as **Not Clicked**. Then this type of category matched with defination of **We did not target a consumer, but consumer clicked on ads,GAINING MONEY 700 CAD**. 


- *True Positive* (**TP**): It means actual class is Clicked **(consumer clicked on Ads)** and correctly classified as **clicked**. Then this type of category matched with defination of **We target a consumer and clicked on ads,GAINING MONEY 150 CAD**. 



- *False Negative* (**FN**): It means actual class is Not Clicked **(consumer did not clicked on Ads)** and incorrectly classified as **Clicked**. Then this type of category matched with defination of  **We target a consumer but the consumer did not clicke on ads,LOSSING MONEY 600 CAD**. 


- *True Negative* (**TN**): It means actual class is Not Clicked **(consumer did not clicked on Ads)** and correctly classified as **Not Clicked**. Then this type of category matched with defination of **We did not target a consumer and the consumer did not click on ads, NOT GAININ or LOSSING MONEY**. 

**Final Summary Results:** 

![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/Summary_Performance.PNG)

Considering test set which has a sample size of 301 customers, Logistic Regression classifier outperforms Baseline, Random Forest and Xgboost Classifier with following metrics: 

- AUC = 97%
- Accuracy = 97% 
- F-Beta Score = 97.97%

Logistic Regression predict **overall investiment return of $25,150 CAD** which includes: 

- Profit gain from True Positive $21,450 CAD

- Profit gain from False Positve $4,900 CAD 

- Loss from False Negative  -$2,400 CAD 

![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/LR_1.PNG)
![alt text](https://github.com/basalem/Optimization_of_Targeted_Advertising_Market_Campaign/blob/master/images/LR_2.PNG)

## Business Recommendation 
The marketing team can maximize client's profit by collecting some information about customers, those features which are proved to be essential for machine learning classifier: 

1. Daily Internet Usage. 
2. Daily Time Spent on Site. 
3. Average Area Income. 
4. Age. 

To further maximize profit, data visualization above shows great insights and correlations with target class, therefore, potential targeted population would be consumers with: 

- Less spending habits on the site less than 60 minutes. 
- Less spending habits in the internet less than 150 minutes.
- Low and Middle Class income less than or equal to $50k CAD. 
- Aged between 30-50 years old. 

Please feel free to reach out for any discussion or feedback! 
