import dill
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
import numpy as np
import scipy as sp

res = requests.get('https://www.patentsview.org/api/assignees/query?q={"_contains":{"assignee_organization":"Adobe"}}&f=["patent_number","app_date","patent_num_claims","detail_desc_length","nber_num_patents_for_assignee","nber_total_num_patents","nber_category_title","patent_num_cited_by_us_patents","patent_num_us_patent_citations","patent_num_combined_citations","patent_date","assignee_organization","assignee_id"]')
patents=res.json()
sym = 'ADBE'
params = {
        'apikey': '{*************}',
        'symbol':sym,
        'outputsize' : 'full'
      }
r2 = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED',params=params)
stocks =r2.json()
#import previously requested data, if not then use requests. 
#patents=dill.load(open("ibm_patents.dill",mode="rb"))
#stocks=dill.load(open("ibm_stocks.dill",mode='rb'))

#We will be looking at the adjusted closing price, aggregated monthly.
df_stocks=pd.DataFrame.from_dict(stocks['Monthly Adjusted Time Series'],orient='index')
df_stocks.index = pd.to_datetime(df_stocks.index)
df_stocks['5. adjusted close']=df_stocks['5. adjusted close'].astype(float)

#There are several groups patent listings with variations on the busuiness name.
#Lets collect those into one bucket.
patents_all=[]
for i in range(len(patents['assignees'])):
    patents_all += patents['assignees'][i]['patents']
#Similarly lets establish a data frame to work with for monthly stock data
   
df_patents=pd.DataFrame(patents_all)
df_patents['patent_date']= pd.to_datetime(df_patents['patent_date'])
df_patents['month_year']=df_patents['patent_date'].dt.to_period('M')
df_patents[["patent_num_claims","detail_desc_length"\
            ,"patent_num_cited_by_us_patents","patent_num_combined_citations"]]\
    =df_patents[["patent_num_claims","detail_desc_length","patent_num_cited_by_us_patents"\
                ,"patent_num_combined_citations"]].apply(pd.to_numeric)
df_patents.index=df_patents['patent_date']

#We want to group these by month and look at patents in the relatively recent past.
df_patents_filtered=df_patents.groupby(['month_year']).sum()
df_patents_filtered=df_patents_filtered.loc[df_patents_filtered.index> '2000-01']

#Aggregated monthly patent features to be used for predictions.
df_patents_filtered.head()

#We shift the patent data set by 3 quarters. The assumption is that new
#technologies are on average introduced every third quarter to annually by tech companies.
df_patents_shifted=df_patents_filtered.shift(periods=9, freq="M")
df_patents_shifted.head()

df_stocks.index=df_stocks.index.to_period("M")
df_patents_shifted.index.rename('index',inplace=True)
df_stocks.index.rename('index',inplace=True)
df = pd.merge(df_stocks, df_patents_shifted, on='index', how='outer').fillna(value=0)

min_date=max(min(df_patents_shifted.index),min(df_stocks.index))


time_series=df['5. adjusted close'].loc[df.index>=min_date].sort_index(ascending=True)
m=time_series.shape[0]-10
patent_features=df[['patent_num_claims',"detail_desc_length","patent_num_cited_by_us_patents"]].loc[df.index>=min_date].sort_index(ascending=True)
#We select the data points to be used for the prediction.
train_X, train_y  = time_series[:m+1],patent_features[:m+1]
remainder_X=time_series[m:]
pred_y  =patent_features[m:]

#The other benefit of pmdrima is that it is simple to implement 
#stationarity. We do so by setting d = 1 in the code below.
sxmodel = auto_arima(train_X, exogenous=pd.DataFrame(train_y),
                           start_p=1, start_q=1,
                           test='adf',d=1,
                           max_p=10, max_q=10, m=4,
                           start_P=0, seasonal=True,
                           information_criterion='aic', 
                            D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True
                           )


#We extract the curve fit from the model to 
#compare to the actual data.
fitted=sxmodel.fit(train_X,pd.DataFrame(train_y))
#Predictions for the final plots

predictions=sxmodel.predict(n_periods=pred_y.shape[0],exogenous=pd.DataFrame(pred_y),return_conf_int=True)
#Upper/lower bounds to establish uncertainities in our computations.
lower=[each[0] for each in predictions[1]]
upper=[each[1] for each in predictions[1]]

#Now we construct the final plots
data={'period':pred_y.index,'val':predictions[0]}
dataLower={'period':pred_y.index,'val':lower}
dataUpper={'period':pred_y.index,'val':upper}
predictions = pd.DataFrame(data)
predictionsLow = pd.DataFrame(dataLower)
predictionsUp = pd.DataFrame(dataUpper)
predictions.index = predictions['period']
predictionsLow.index = predictionsLow['period']
predictionsUp.index = predictionsUp['period']

fitted=sxmodel.predict_in_sample(pd.DataFrame(train_y))

#predictions.index = predictions['period']
#predictionsLow.index = predictionsLow['period']
#predictionsUp.index = predictionsUp['period']
plt.figure(figsize=(10,8), dpi=400)
train_X[150:].plot(label='Actual Data',color='cornflowerblue')
remainder_X.loc[remainder_X.index<=max(df_stocks.index)].plot(label='Actual Test Data',color='cornflowerblue',linestyle='--', alpha=0.8)
fitted[150:].shift(-1).plot(label='Fitted Model',color='y')
predictions['val'].plot(label='Predictions',color='black')
plt.fill_between(predictionsUp.index, 
                 predictionsUp['val'], 
                 predictionsLow['val'], 
                 color='gray', alpha=.10)
predictionsUp['val'].plot(label='95% Confidence Interval',linestyle='--', linewidth=1, alpha=1,color='gray')
predictionsLow['val'].plot(label='95% Confidence Interval',linestyle='--', linewidth=1, alpha=1,color='gray')
plt.title("Predicted vs Actual Stock Prices for {}".format(sym),fontsize=22)
plt.xlabel("Time(Month-Year)",fontsize=22)
plt.ylabel("Adjusted Closing Stock Price (USD)",fontsize=22)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=16)
plt.legend(fontsize=13)
plt.ylim(min(train_X[150:])*0.75,max(train_X[150:])*1.3)
plt.plot()
plt.savefig("prediction_{}.png".format(sym))

