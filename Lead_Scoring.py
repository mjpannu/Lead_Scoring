#!/usr/bin/env python
# coding: utf-8

# ### Importing Dataset

# In[1]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing Pandas and NumPy
import pandas as pd, numpy as np


# In[3]:


# Importing all datasets
leads_data = pd.read_csv("Leads.csv")
leads_data.head()


# In[4]:


leads_data.shape


# In[5]:


leads_data.describe()


# In[6]:


leads_data.info()


# #### Converting some binary variables (Yes/No) to 0/1

# In[7]:


# List of variables to map

varlist =  ['Do Not Email' ,'Do Not Call', 'Search','Magazine','Newspaper Article','X Education Forums',
            'Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
           'Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
leads_data[varlist] = leads_data[varlist].apply(binary_map)




# In[8]:


leads_data.head()


# ### Handling Select value in columns 

# In[9]:


print(leads_data['Specialization'].isnull().sum())
print(leads_data['How did you hear about X Education'].isnull().sum())
print(leads_data['Lead Profile'].isnull().sum())
print(leads_data['City'].isnull().sum())
print("-------------")
print(leads_data['Specialization'].apply(lambda x : (None if x == 'Select' else x)).isnull().sum())
print(leads_data['How did you hear about X Education'].apply(lambda x : (None if x == 'Select' else x)).isnull().sum())
print(leads_data['Lead Profile'].apply(lambda x : (None if x == 'Select' else x)).isnull().sum())
print(leads_data['City'].apply(lambda x : (None if x == 'Select' else x)).isnull().sum())


# In[10]:


leads_data['Specialization'] = leads_data['Specialization'].apply(lambda x : (None if x == 'Select' else x))
leads_data['How did you hear about X Education'] = leads_data['How did you hear about X Education'].apply(lambda x : (None if x == 'Select' else x))
leads_data['Lead Profile'] = leads_data['Lead Profile'].apply(lambda x : (None if x == 'Select' else x))
leads_data['City'] = leads_data['City'].apply(lambda x : (None if x == 'Select' else x))


# In[11]:


leads_data.isnull().sum()


# In[12]:


###Bucket all the less frequent categories of categorical variables

def handle_leads(x):
    if x in ('Quick Add Form','Lead Add Form','Lead Import'):
        return 'LessFrequent'
    else:
        return x
    
leads_data['Lead Origin'] = leads_data['Lead Origin'].apply(handle_leads)

def handle_lead_source(x):
        if x in ['Referral Sites', 'Reference','google', 'Welingak Website',
           'Facebook', 'blog', 'Pay per Click Ads', 'bing', 'Social Media',
           'WeLearn', 'Click2call', 'Live Chat', 'welearnblog_Home',
           'youtubechannel', 'testone', 'Press_Release', 'NC_EDM']:
            return 'LessFrequent'
        #'Referral Sites', 'Reference'
        else:
            return x

leads_data['Lead Source'] = leads_data['Lead Source'].apply(handle_lead_source)

def handle_Last_Notable_Activity(x):
    #Olark Chat Conversation
        if x in ['Approached upfront','Email Bounced','Email Link Clicked','Email Marked Spam','Email Received','Form Submitted on Website','Had a Phone Conversation','Resubscribed to emails','Unreachable','Unsubscribed','View in browser link Clicked']:
            return 'LessFrequent'
        else:
            return x

leads_data['Last Notable Activity'] = leads_data['Last Notable Activity'].apply(handle_Last_Notable_Activity)

def handle_last_Activity(x):
        if x in ['Email Marked Spam','Email Received','Form Submitted on Website','Had a Phone Conversation','Resubscribed to emails','Unreachable','Unsubscribed','View in browser link Clicked',
                 'Visited Booth in Tradeshow']:
            return 'LessFrequent'
        else:
            return x

leads_data['Last Activity'] = leads_data['Last Activity'].apply(handle_last_Activity)


# In[13]:


#round(100*(leads_data.isnull().sum()/len(leads_data.index)), 2)
leads_data['Last Notable Activity'].value_counts()


# ### Dropping columns with missing value more than 3000 and less relavancy

# In[14]:


leads_data=leads_data.drop(['Country','Specialization','How did you hear about X Education','What is your current occupation','What matters most to you in choosing a course','Tags','Lead Quality','Lead Profile','City','Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score','Asymmetrique Profile Score'],1)


# In[15]:


leads_data.dropna(inplace = True)


# In[16]:


#leads_data.dropna(subset=['Lead Source','TotalVisits'], inplace=True)

leads_data.isnull().sum()
leads_data.head()


# #### For categorical variables with multiple levels, create dummy features (one-hot encoded)

# In[17]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(leads_data[['Lead Origin','Lead Source','Last Activity','Last Notable Activity']], drop_first=True)

# Adding the results to the master dataframe
leads_data = pd.concat([leads_data, dummy1], axis=1)

leads_data = leads_data.drop(['Lead Origin','Lead Source','Last Activity','Last Notable Activity'],1)


# In[18]:


leads_data.info()


# ### Test-Train Split

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


# Putting feature variable to X
X = leads_data.drop(['Prospect ID','Lead Number','Converted'], axis=1)

X.head()


# In[21]:


# Putting response variable to y
y = leads_data['Converted']

y.head()


# In[55]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ### Feature Scaling

# In[56]:


from sklearn.preprocessing import StandardScaler


# In[57]:


scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()
X_train = X_train.fillna(X_train.mean())
X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']].describe()


# In[58]:


### Checking the Churn Rate
conversion = (sum(leads_data['Converted'])/len(leads_data['Converted'].index))*100
conversion


# In[59]:


# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(leads_data.corr(),annot = True)
plt.show()


# ### Model Building

# In[60]:


import statsmodels.api as sm


# In[61]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# ### Feature Selection Using RFE

# In[86]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[87]:


from sklearn.feature_selection import RFE
rfe = RFE(estimator=logreg, n_features_to_select=20)             # running RFE
#rfe = RFE(logreg, 50)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)


# In[88]:


rfe.support_


# In[89]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[90]:


col = X_train.columns[rfe.support_]


# In[91]:


X_train.columns[~rfe.support_]


# In[92]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[93]:


y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[94]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[95]:


y_train_pred_final = pd.DataFrame({'Conversion':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final['CustID'] = y_train.index
y_train_pred_final.head()


# In[96]:


y_train_pred_final['predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[97]:


from sklearn import metrics


# In[98]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Conversion, y_train_pred_final.predicted )
print(confusion)


# In[99]:


print(metrics.accuracy_score(y_train_pred_final.Conversion, y_train_pred_final.predicted))


# #### Checking VIFs

# In[100]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[101]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[102]:


# Dropping columns multiple times as per VIF and p values

col = col.drop(['Last Activity_LessFrequent','Lead Origin_LessFrequent','Do Not Call', 'Newspaper'
               ,'Through Recommendations','Last Activity_SMS Sent','Newspaper Article', 'X Education Forums'], 1)
col


# In[103]:


X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[104]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final['Conversion_Prob'] = y_train_pred
y_train_pred_final['predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
print(metrics.accuracy_score(y_train_pred_final.Conversion, y_train_pred_final.predicted))


# In[105]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[106]:


y_train_pred_final['Lead_score'] = y_train_pred_final['Conversion_Prob']*100


# In[107]:


y_train_pred_final.describe()


# In[108]:


y_train_pred_final.head()


# In[109]:


print(metrics.accuracy_score(y_train_pred_final.Conversion, y_train_pred_final.predicted))


# In[110]:


X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])


# In[111]:


X_test = X_test[col]
X_test.head()


# In[112]:


X_test_sm = sm.add_constant(X_test)


# In[113]:


y_test_pred = res.predict(X_test_sm)


# In[114]:


y_test_pred[:10]


# In[115]:


y_pred_1 = pd.DataFrame(y_test_pred)


# In[116]:


y_pred_1.head()


# In[152]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[153]:


y_test_df


# In[154]:


y_test_df['CustID'] = y_test_df.index


# In[155]:


y_test_df


# In[156]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[157]:


y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[158]:


y_pred_final= y_pred_final.rename(columns={ 0 : 'Conversion_Prob'})



# In[159]:


y_pred_final.head()


# In[173]:


y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[174]:


y_pred_final.head()


# In[175]:


metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)


# In[ ]:




