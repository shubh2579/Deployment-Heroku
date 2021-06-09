#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


df = pd.read_csv('CarPrice_Assignment.csv')


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


df.head()


# In[8]:


print("No of Car Name :" + str(len(df['CarName'].unique())))


# In[164]:


# Lets ignore CarName
df['enginetype'].unique()


# In[10]:


df['price'].plot()


# In[11]:


df.drop(['CarName','car_ID'] , inplace = True, axis = 1)
df.shape


# In[12]:


import matplotlib.pyplot as plt 
import seaborn as sns 

plt.figure(figsize=(15,10))
sns.heatmap(df.corr() , cmap="YlGnBu",annot=True)


# In[13]:


data = pd.get_dummies(df, drop_first =True)
data.head()


# In[14]:


data.columns


# In[15]:


df.corr()


# In[140]:


X = data.drop('price' , axis = 1)
y = df['price']
X.shape


# In[141]:


from sklearn.feature_selection import mutual_info_regression
miv=mutual_info_regression(X,y)
#mutual_data=pd.Series(miv,index=X.columns)
type(miv)
#mutual_data.sort_values(ascending=False)
#mutual_data
miv
new_value=[]
for name,i in zip(X.columns,miv):
    if i==0:
        new_value.append(name)
new_value


# In[142]:


X_new=X[['doornumber_two',
 'carbody_sedan',
 'enginetype_dohcv',
 'cylindernumber_twelve',
 'fuelsystem_4bbl']]
X_new.shape


# In[137]:


#from sklearn.feature_selection import SelectKBest,f_regression

#Selector=SelectKBest(f_regression,k=5)
#X_new=Selector.fit_transform(X,y)
#X_new.get_support()

#X_new=pd.DataFrame(Selector)



#X_new.shape

#X_new=pd.DataFrame(X_new,columns=X_new.names)
#mask = X_new.get_support() #list of booleans
#new_features = [] # The list of your K best features

#for bool, feature in zip(mask, feature_names):
    #if bool:
       ## new_features.append(feature)


# In[143]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( X_new,y , test_size = 0.3, random_state = 101) 
print(X_train.shape)
print(X_test.shape)


# In[144]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)


# In[145]:


reg.score(X_train, y_train)


# In[146]:


reg.coef_


# In[147]:


reg.intercept_


# In[148]:


y_pred = reg.predict(X_test)


# In[149]:


from sklearn.metrics import mean_squared_error

MSE  = mean_squared_error(y_test, y_pred)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)


# In[150]:


from sklearn.metrics import r2_score
from sklearn.metrics import classification_report

r2 = r2_score(y_test, y_pred)

print("R2 :" ,r2)


# ## Lasso Regression

# In[151]:


from sklearn import linear_model
lasso  = linear_model.Lasso(alpha=1 , max_iter= 3000)

lasso.fit(X_train, y_train)


# In[152]:


lasso.score(X_train, y_train)


# In[153]:


y_pred_l = lasso.predict(X_test)


# In[154]:


MSE  = mean_squared_error(y_test, y_pred_l)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)

r2 = r2_score(y_test, y_pred_l)
print("R2 :" ,r2)


# ## Rigde regression

# In[155]:


from sklearn.linear_model import Ridge

ridge  = Ridge(alpha=0.1)


# In[156]:


ridge.fit(X_train,y_train)


# In[157]:


ridge.score(X_train, y_train)


# In[158]:


y_pred_r = ridge.predict(X_test)


# In[159]:


MSE  = mean_squared_error(y_test, y_pred_r)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)

r2 = r2_score(y_test, y_pred_r)
print("R2 :" ,r2)


# In[160]:


import pickle


# In[161]:


with open('model_pickle','wb') as file:
    pickle.dump(lasso,file)


# In[162]:


with open('model_pickle','rb') as file:
    lasso_model=pickle.load(file)
    


# In[163]:


lasso.predict(X_test)


# In[ ]:




