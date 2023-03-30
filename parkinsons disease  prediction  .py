#!/usr/bin/env python
# coding: utf-8

# # Data:-
# * The original data taken from the  UCI machine learning repository https://archive.ics.uci.edu/ml/datasets/parkinsons

# # Data Set Information
# * This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). 
# * Each column in the table is a particular voice measure, and each row corresponds one of 195 voice recording from these individuals ("name" column).
# * The rows of the CSV file contain an instance corresponding to one voice recording. There are around six recordings per patient, the name of the patient is identified in the first column.

# # The main aim of the data is to classify healthy people from those with PD.
# 

# # According to "status" column which is set to 0 for healthy and 1 for PD.

# # Attribute Information:
# 
# * Matrix column entries (attributes):
# * name - ASCII subject name and recording number
# * MDVP:Fo(Hz) - Average vocal fundamental frequency
# * MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
# * MDVP:Flo(Hz) - Minimum vocal fundamental frequency
# * MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several measures of variation in fundamental frequency
# * MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
# * NHR,HNR - Two measures of ratio of noise to tonal components in the voice
# * status - Health status of the subject (one) - Parkinson's, (zero) - healthy
# * RPDE,D2 - Two nonlinear dynamical complexity measures
# * DFA - Signal fractal scaling exponent
# * spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation
# 
# 

# In[1]:


import sklearn.datasets
import warnings
warnings.filterwarnings
import sklearn

import pandas as pd
import numpy as np

# To visualize the data Which  help the reader to achieve quick insights.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_theme()


#To standardize the data
from sklearn.preprocessing import StandardScaler

# classification models from Scikit-Learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC


## To Evaluate the Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import StandardScaler


# # Load data

# In[2]:


#loading the data
parkinsons =pd.read_csv('parkinsons.csv')

parkinsons.head()
    


# In[3]:


# number of rows and coloumns
parkinsons.shape


# # Data Exploration (exploratory data analysis or EDA)

# In[4]:


# getting more information
parkinsons.info()


# # checking for missing values

# In[5]:


parkinsons.isnull().sum()


# In[6]:


# getting some statistical measurements about the data
parkinsons.describe()


# # Let's find out how many of each class there

# In[7]:


# distributation of tarhet varible
parkinsons['status'].value_counts()


# In[8]:


#  0---->no parkinsons
#  1----> parkinsons


# In[9]:


parkinsons["status"].value_counts().plot(kind="bar",color=["red", "blue"])
plt.xlabel("1 = With PD,            0 = HEALTHY")
plt.ylabel("Amount")
plt.legend()


# # checking the distribution of the spread1 column with a histogram

# In[10]:


parkinsons.spread1.plot.hist();


# In[11]:


parkinsons.spread2.plot.hist();


# # Make a Correlation matrix

# In[12]:


plt.figure(figsize=(15,10))
corr=parkinsons.corr()
sns.heatmap(corr,annot=True,annot_kws={'size':8},cmap='Blues')


# # Modelling

# In[13]:


#grouping the data based on the target varible
parkinsons.groupby('status').mean()


# In[14]:


#data preprocessing
# separating the features and target
x = parkinsons.drop(columns=['name','status'],axis=1)
y = parkinsons['status']


# In[15]:


print(y)


# In[16]:


# splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


# In[17]:


print(x.shape, x_train.shape, x_test.shape)


# In[18]:


# data satandardization
scaler = StandardScaler()
scaler.fit(x_train)


# In[19]:


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[20]:


print(x_train)


# # Now we've got our data split into train and test sets,it's time to build a machine learning model

# # Model training 

# # support vector machine model

# In[21]:


model = svm.SVC()
model.fit(x_train, y_train)# training  the svm model with training data


# In[22]:


# accuracy score on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[23]:


# accuracy score on testing data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# # LogisticRegression

# In[24]:


model = LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# # RandomForestClassifier

# In[25]:


model = RandomForestClassifier()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# # Support Vector Classification

# In[26]:


clf = SVC()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)


# # Evaluating our tuned machine learning classifier, beyond accuracy
# * ROC curve and AUC score
# * confusion matrix
# * classfication report
# * precision
# * recall
# * F1-score

# # Plot ROC curve and calculate and calculate AUC metrics

# In[27]:


plot_roc_curve(clf, x_test, y_test)


# # confusion_matrix

# In[28]:


print(confusion_matrix(y_test,x_test_prediction))


# In[29]:


sns.set(font_scale=1.5)

def plot_conf_mat(y_test,x_test_prediction ):
    
    """
    plot a nice looking confusion matrix using seaborn's heatmap()
    """
    
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test, x_test_prediction),
                     annot=True,
                     cbar=False)
    plt.xlabel("True label")
    plt.ylabel("predicted label")
    
   
    
plot_conf_mat(y_test, x_test_prediction)


# # Now lets get a classification report as well as cross-validation precision and f1-score

# In[30]:


print(classification_report(y_test,x_test_prediction ))


# In[28]:


#Building a prediction systerm


# In[78]:


input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)
#change input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)

if(prediction[0]==0):
    print("The person does not have parkinsons Disease")
    
else:
    print("The person having parkinsons Disease")


# # Downloding the Model

# In[30]:


import pickle
filename = 'trained_model.sav'
pickle.dump(model,open(filename,'wb'))


# In[31]:


#load the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))


# In[32]:


input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)
#change input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = loaded_model.predict(std_data)
print(prediction)

if(prediction[0]==0):
    print("The person does not have parkinsons Disease")
    
else:
    print("The person having parkinsons Disease")


# In[ ]:





# In[ ]:





# In[ ]:




