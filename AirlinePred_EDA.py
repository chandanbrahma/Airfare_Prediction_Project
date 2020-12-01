


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sb
from sklearn.model_selection import train_test_split



data=pd.read_csv(r"C:\\datascience\\Airfare Prediction\\Concatenate_B2C_B2E.csv")

data.head()

data.shape

df=data.iloc[:,:4]

df.head()

sb.countplot(x='ProductType',data=df)

df.columns

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.head()

df['ItineraryType'].value_counts()


#cols = df.columns.difference(['InvoiceDate'])
#if mixed dtypes
#df[cols] = df[cols].astype(str).astype(float)


#df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

#df_n = df.set_index('InvoiceDate').groupby(pd.Grouper(freq='d')).mean().dropna(how='all')
#print (df_n)


# There are 425 observations which means for 425 days if we take the mean 


df['Month']= pd.DatetimeIndex(df['InvoiceDate']).month
df['Day']= pd.DatetimeIndex(df['InvoiceDate']).day
df['Year'] = pd.DatetimeIndex(df['InvoiceDate']).year
df['hour'] = pd.DatetimeIndex(df['InvoiceDate']).hour
df['weekday'] = pd.DatetimeIndex(df['InvoiceDate']).weekday


df=df.drop('InvoiceDate',axis=1)

df.head()

df.dtypes


df.isnull().sum()


##As we see index no 180141 and 180143 contains all the null values, So dropping the rows

df=df.drop([180141, 180143])
df.isnull().sum()

##Checking the null values for the 'ItineraryType' column
df[df.ItineraryType.isnull()]

df.loc[df.ProductType=='Charge','ProductType'].count()


# ##Since all the charges are zero in the data as well as theree is no mention of iternity type, international or domestic. These variables are not giving any weightage to our data, Hence we will eliminate them

df=df[df.ProductType != 'Charge']


##Now lets check the null values for the 'ItineraryType' column
df[df.NetFare.isnull()]

##from the net fare it seems that the null value is for the payment and refund Product type
##lets count the numbers of payment and refund for our confirmtion
df.loc[df.ProductType=='payment','ProductType'].count()
df.loc[df.ProductType=='refund','ProductType'].count()


## So as we found out that the total number of null values in the Netfare column is of the payment and refund producttype only
## It means we an assume that the company has not refunded any amout .So imputing those space with 0
df['NetFare']=df['NetFare'].fillna(0)

df.isna().sum()

##checking the datatype of the columns
df.dtypes
df['NetFare']=df['NetFare'].astype(float)


## checking the unique varibles of different columns
df['ProductType'].unique()

df['ItineraryType'].unique()

sb.heatmap(df.corr(),annot=True)


#Correlation
dummies = pd.DataFrame(pd.get_dummies(df, columns=["ProductType"]))
dummies = pd.DataFrame(pd.get_dummies(dummies, columns=["ItineraryType"]))
dummies['NetFare'] = pd.to_numeric(dummies['NetFare'] ,errors='coerce')
a=pd.DataFrame.corr(dummies)
a


# There is a low correlation of 0.29 between Netfare and Air which is significantly higher than other product Types. Also Netfare shows significantly moderate correlation with Intl internity type

# ## EDA for Product-Air and Iternity Type-Domestic  

#considering only air and domestic type 
df1=df[df['ProductType']=='Air']
df_dom=df1[df1['ItineraryType']=='Domestic']


df_dom.head()


df_dom.dtypes


sb.distplot(df_dom.NetFare) 


import statsmodels.api as sm
sm.qqplot(df_dom['NetFare'],line='s')


plt.hist(df_dom['NetFare'])



# ### Shapiro-wilk Test


from scipy.stats import shapiro
# normality test
stat, p = shapiro(df_dom['NetFare'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')



sb.countplot(x='Month',data=df_dom)
#df_dom.describe()


# Highest Airtravel happened in the month of May whereas comparatively less airtravel in June - August obviously because of monsoon season whereas it stars picking up from September


sb.countplot(x='Year',data=df_dom)


sb.countplot(x='Day',data=df_dom)


sb.countplot(x='weekday',data=df_dom)



sb.countplot(x='hour',data=df_dom)


# Increase in counts in the daytime starting from 7am in the morning



#plot of 31 days' avg netfare  
h=[]
r=range(1,30)
for i in r:
    d=df_dom[df_dom['Day']==i]
    Avg=np.mean(d['NetFare'])
    h.append(Avg)
    
plt.plot(r,h)


# Highest average Netfare is noticed on 12th day of the month


#plot of 12 months' avg netfare  
j=[]
r=range(1,13)
for i in r:
    d=df_dom[df_dom['Month']==i]
    Avg=np.mean(d['NetFare'])
    j.append(Avg)
    
plt.plot(r,j)


# Hike in fare obeserved in the month of April and Nov Due to summer holidays and festival


#plot of 24 hrs' avg netfare  
k=[]
r=range(0,24)
for i in range(0,24):
    Hrs=df_dom[df_dom['hour']==i]
    Avg=np.mean(Hrs['NetFare'])
    k.append(Avg)
plt.plot(r,k)


# Low Avg netfare observed during night, it starts increasing from early morning. Post 20:00 , there is again a drop in the netfare



x.weekday.value_counts()


#plot of week days' avg netfare  
l=[]
r=range(0,7)
for i in range(0,7):
    Week=df_dom[df_dom['weekday']==i]
    Avg=np.mean(Week['NetFare'])
    l.append(Avg)
plt.plot(r,l)
#low price on thursdays and weekends


# Netfare starts dropping from Tuesday and starts picking up from Wednesday and again drops from Friday which is surprising
# 

# # EDA for Product type-Air and Iternity Type-International


df2=df[df['ProductType']=='Air']
a=df2[df2['ItineraryType']=='International']
a.head()


a.columns

df.ItineraryType.value_counts()

sb.distplot(a.NetFare) 

import statsmodels.api as sm
sm.qqplot(a['NetFare'],line='s')


from scipy.stats import shapiro
# normality test
stat, p = shapiro(a['NetFare'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')


import seaborn as sb
sb.countplot(x='Month',data=a)


# High count in the month of May mainly due to Summer vacation also we can see increasing trend in the month of Feb and  March


sb.countplot(x='Year',data=a)


sb.countplot(x='Day',data=a)


sb.countplot(x='weekday',data=a)


#for col in df.columns:
#    df[col][df[col] < 0] = 0



temp= df_dom



temp.head()



x = temp.drop('NetFare',axis=1)
y = temp['NetFare']

x.head()
#y.head()

df.columns
df.ProductType.value_counts()


df['ProductType']=pd.to_numeric(df['ProductType'], errors='coerce').fillna(0, downcast='infer')



# Relation of Netfare and Producttype by Month
fig, ax = plt.subplots(figsize=(15,7))

count_prodtype=df.groupby(['Month','ProductType']).count()['NetFare'].unstack().plot(ax=ax)



# Relation of Netfare and Producttype by Day
fig, ax = plt.subplots(figsize=(15,7))

count_prodtype=df.groupby(['Day','ProductType']).count()['NetFare'].unstack().plot(ax=ax)


# Relation of Netfare and Producttype by Day
fig, ax = plt.subplots(figsize=(15,7))

count_prodtype=df.groupby(['Year','ProductType']).count()['NetFare'].unstack().plot(ax=ax)


# Relation of Netfare and Producttype by weekday
fig, ax = plt.subplots(figsize=(15,7))

count_prodtype=df.groupby(['weekday','ProductType']).count()['NetFare'].unstack().plot(ax=ax)


# Relation of Netfare and Producttype by hour
fig, ax = plt.subplots(figsize=(15,7))

count_prodtype=df.groupby(['hour','ProductType']).count()['NetFare'].unstack().plot(ax=ax)


# Relation of Netfare and Iternitytype by Month
fig, ax = plt.subplots(figsize=(15,7))

count_prodtype=df.groupby(['Month','ItineraryType']).count()['NetFare'].unstack().plot(ax=ax)


# Trend of travel for both Domestic and Intl is almost same in the month of May i.e during vacations it's on higher side. Domestic Netfare shows a slight increase during Oct/November i.e festival season whereas INTL fares remains constant


# Relation of Netfare and Iternitytype by weekday
fig, ax = plt.subplots(figsize=(15,7))

count_prodtype=df.groupby(['weekday','ItineraryType']).count()['NetFare'].unstack().plot(ax=ax)


# There are no fluctuations in the netfare for Intl whereas for Domestic we can see drop in the fare on Monday and starts picking up therefater. From Thursday , fares again starts dropping over the weekend



# Relation of Netfare and Iternitytype by hour
fig, ax = plt.subplots(figsize=(15,7))

count_prodtype=df.groupby(['hour','ItineraryType']).count()['NetFare'].unstack().plot(ax=ax)


# For Domestic, shows upward trend during day time from 6am till evening 17:00 post which shows a drop in the count. It shows ppl prefer to travel during day time. For INTL, shows a constant trend except for a slight incrase from 7am in the morning



df_n = df.drop('InvoiceDate',axis=1)


df_n.columns



df_n.dtypes
df.columns
df.shape


###Duplicates
df_duplicate=df.duplicated(subset=['InvoiceDate', 'NetFare', 'ProductType', 'ItineraryType', 'Month',
       'Day', 'Year', 'hour', 'weekday'], keep='first')

df_duplicate.value_counts()

###Check this
#df_n['NetFare'] = pd.to_numeric(df_n['NetFare'])


num = df_n._get_numeric_data()
num
#num[num < 0] = 0


num.isnull().sum()


# ## Linear Regression


df.columns


from statsmodels.formula.api import ols

model = ols('NetFare ~ C(ProductType)', df).fit()
model.summary()



import statsmodels.api as sm
# Seeing if the overall model is significant
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

# Creates the ANOVA table
res = sm.stats.anova_lm(model, typ= 2)
res


# P value is 0.000. The true p-value is not 0.000. Actual p value is less than 0.0005 and using bulit in rounding rules it is rounded down and reported as 0.000. Typically it is reported as p<.001.
# p of 0.0000 means reasults are highly significant , so ProductType is a significant variable has overall significant effect on Netfare



###Let's Check the variable Iternity Type
model2 = ols('NetFare ~ C(ItineraryType)', df).fit()
model2.summary()



import statsmodels.api as sm
# Seeing if the overall model is significant
print(f"Overall model F({model2.df_model: .0f},{model2.df_resid: .0f}) = {model2.fvalue: .3f}, p = {model2.f_pvalue: .4f}")

# Creates the ANOVA table
res = sm.stats.anova_lm(model2, typ= 2)
res


# It shows that ItineraryType is statistically significant



###Let's Check the variable Iternity Type
#model3 = ols('NetFare ~ C(ProductType) * C(ItineraryType)*C(Month)*C(Day)*C(Year)*C(hour)*C(weekday)', df).fit()
#model3.summary()
#import statsmodels.api as sm
## Seeing if the overall model is significant
#print(f"Overall model F({model3.df_model: .0f},{model3.df_resid: .0f}) = {model3.fvalue: .3f}, p = {model3.f_pvalue: .4f}")

## Creates the ANOVA table
#res = sm.stats.anova_lm(model3, typ= 2)
#res



import pathlib
pathlib.Path().absolute()


# # Domestic


data=pd.read_csv(r"D:\R Excel Sessions\Projects\Airline Predictions\dom_air.csv")


data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['Day'] = pd.DatetimeIndex(data['InvoiceDate']).day
data['Month'] = pd.DatetimeIndex(data['InvoiceDate']).month



data.head()
#####Daywise pivot for Netfare 
heatmap_y_day= pd.pivot_table(data=data,values="NetFare",columns="Day",aggfunc="mean",fill_value=0)
heatmap_y_day


import seaborn as sns
sns.heatmap(heatmap_y_day,annot=True,fmt="g")


sns.boxplot(x="Day",y="NetFare",data=data)


sns.lineplot(x="Day",y="NetFare",data=data)


#####Monthwise pivot for Netfare 
heatmap_y_month= pd.pivot_table(data=data,values="NetFare",columns="Month",aggfunc="mean",fill_value=0)
heatmap_y_month


sns.heatmap(heatmap_y_month,annot=True,fmt="g")


sns.boxplot(x="Month",y="NetFare",data=data)


sns.lineplot(x="Month",y="NetFare",data=data)


####Check for Stationarity
from statsmodels.tsa.stattools import adfuller
def adf_test(series):    
    result = adfuller(series.dropna())  
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    #for key,val in result[4].items():
        #out[f'critical value ({key})']=val
    if result[1] <= 0.05:
        print("Reject the null hypothesis")
        print("Data is stationary")
    else:
        print("Fail to reject the null hypothesis")
        print("Data is non-stationary")

adf_test(data['NetFare'])


# Data is stationary


# # International


data=pd.read_csv(r"D:\R Excel Sessions\Projects\Airline Predictions\Concatenate_B2C_B2E.csv")


d=data

d=d[d['ItineraryType']=='International']
d.shape


data['NetFare'] = data['NetFare'].apply(pd.to_numeric, errors='coerce')


##There are 20,458 records for INTL
d['NetFare'] = d['NetFare'].apply(pd.to_numeric, errors='coerce')

d.NetFare.plot()

df_orig = pd.read_csv("D:\\R Excel Sessions\\Projects\\Airline Predictions\Pooja\\international.csv",parse_dates=['InvoiceDate'])
df =df_orig.copy()   #creating a copy of it
df.columns
df.head(10)
df.info()
df.NetFare.plot()
df.shape


####DickyFuller Test

adf_test(data['NetFare'])


##Data is stationary


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Day'] = pd.DatetimeIndex(df['InvoiceDate']).day
df['Month'] = pd.DatetimeIndex(df['InvoiceDate']).month


#####Daywise pivot for Netfare 
df.head()
df.shape
heatmap_y_day= pd.pivot_table(data=df,values="NetFare",columns="Day",aggfunc="mean",fill_value=0)
heatmap_y_day


sns.heatmap(heatmap_y_day,annot=True,fmt="g")


sns.boxplot(x="Day",y="NetFare",data=df)


sns.lineplot(x="Day",y="NetFare",data=df)

#####Monthwise pivot for Netfare 
heatmap_y_month= pd.pivot_table(data=df,values="NetFare",columns="Month",aggfunc="mean",fill_value=0)
heatmap_y_month


sns.heatmap(heatmap_y_month,annot=True,fmt="g")


sns.boxplot(x="Month",y="NetFare",data=df)


sns.lineplot(x="Month",y="NetFare",data=df)


import pathlib
pathlib.Path().absolute()
