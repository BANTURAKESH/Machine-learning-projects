#!/usr/bin/env python
# coding: utf-8

# # predicting heart disease using machine learning
# 
# This notebook looks into using various python-based machine learning and data science in an attempt build a machine learning model capalbe of predicting whether or not someone has heart disease based on medical attributes
# 
# 
# we are going to take the following approach:
# 
# 1.Problem definition
# 2.Data
# 3.Evaluation
# 4.Features
# 5.Modelling
# 6.Experimentation
# 
# ## 1.problem defination
# > Given clinical parameters about a patient, can we predict wether or not they have heart disease?
# 
# ## 2. Data
# 
# The original data came from the Cleavland data from the UCI machine learning repository.
# https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
# 
# 
# 
# there is also a version of it available on the Kaggle. https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# 
# 
# ## 3.Evaluation
# 
# >If we can reach 95% accuracy at the predicting wether or not a patient has heart disease during the proof of concept, we'll pursue the project.
# 
# 
# ## 4.Features
# 
# This is wheare you'll get different information about each of the  features in your data.
# 
#  **create data dictionary**
# * age
# * sex
# * chest pain type (4 values)
# * resting blood pressure
# * serum cholestoral in mg/dl
# * fasting blood sugar > 120 mg/dl
# * resting electrocardiographic results (values 0,1,2)
# * maximum heart rate achieved
# * exercise induced angina
# * oldpeak = ST depression induced by exercise relative to rest
# * the slope of the peak exercise ST segment
# * number of major vessels (0-3) colored by flourosopy
# * thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
# * age
# 

# # preparing the tools 
# 
# We are going to use pandas, Matplotlib,Numpy,seaborn,scikit-learn for data analysis and manipulation. 

# In[72]:


## import all the tools we need 

# Regular EDA (exploratory data analysis) and plotting libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# we want our plot to apper inside the note book 

get_ipython().run_line_magic('matplotlib', 'inline')

# models from Scikit-Learn

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
## Model Evaluations

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import StandardScaler


# # Load data

# In[5]:


df = pd.read_csv("heart.csv")
df.head()


# In[6]:


df.shape


# # Data Exploration (exploratory data analysis or EDA)
# 
# The goal here is to find out more about the data and become a subject matter export on the dataset you're working with.
# 
# 1. what questions are you trying to solve?
# 2. what kind of data do we have and how do we treat different types?
# 3. what's missing from the data how do you deal with it?
# 4. where are the outliers and why should you care about them?
# 5. How can you add,change or remove features to get more out of your data?

# In[7]:


df.head(n=10)


# In[8]:


df.tail()


# # Let's find out how many of each class there

# In[9]:


df["target"].value_counts()

0----->no disease
1----->disease present
# In[10]:


df["target"].value_counts().plot(kind="bar",color=["salmon", "blue"])


# In[11]:


df.info()


# # Finding missing values

# In[12]:


df.isna().sum()


# In[13]:


df.describe()


# # Heart disease Frequency according to sex

# In[14]:


df.sex.value_counts()

1---->male
0---->female
# # compare target column with sex colunm

# In[15]:


pd.crosstab(df.target,df.sex)


# In[16]:


sns.catplot(data=df, kind="bar", x="sex", y="target")


# In[17]:


## creating a plot of crosstab
pd.crosstab(df.target,df.sex).plot(kind="bar",figsize=(10,6),color=["salmon","lightblue"])

plt.title("Heart disease Frequency for sex")
plt.xlabel("0 = no heart disease,1 = didease")
plt.ylabel("Amount")
plt.legend(["female","male"])


# # checking the distribution of the age column with a histogram
# 

# In[18]:


df.age.plot.hist();


# # Heart disease frequency per chest pain Type 
# *  cp: chest pain type
# 0. Value 1: typical angina
# 1. Value 2: atypical angina
# 2. Value 3: non-anginal pain
# 3. Value 4: asymptomatic

# In[19]:


pd.crosstab(df.cp,df.target)


# # Making the crosstab more visual

# In[20]:


pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(10,6),color=["salmon","lightblue"])

## Adding some information
plt.title("Heart disease Frequency per chest pain type")
plt.xlabel("cheat pain type")
plt.ylabel("Amount")
plt.legend(["No disease","Disease"])
plt.xticks(rotation=0);


# In[21]:


df.head()


# # Make a Correlation matrix

# In[22]:


df.corr()


# # Making a correlation matrix [Heat map]  

# In[23]:


plt.figure(figsize=(10,10))
corr=df.corr()
sns.heatmap(corr,annot=True,annot_kws={'size':8},cmap='Blues')


# # 5.Modelling

# In[24]:


df.groupby('target').mean()


# In[25]:


# split the data into x and y
x = df.drop("target",axis=1)

y = df["target"]


# In[26]:


x


# In[27]:


y


# In[28]:


#split data into train and test sets
np.random.seed(42)

#split the data in to train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[29]:


x_train


# In[30]:


y_train,len(y_train)


# In[31]:


x_test


# In[32]:


y_test


# In[33]:


print(x.shape, x_train.shape, x_test.shape)


# # Now we've got our data split into train and test sets,it's time to build a machine learning model
# 
# we'll train it(find the patterns)on the train set.
# 
# And we'll test it(use the patterns) on the test set 

# # we're going to try 3 different machine learning models 
# 1. K- Nearest Neighbours Classifier
# 2. Logistic Regression
# 3. Random Forest classifier

# In[34]:


#put models in a dictionary
models = {"Logistic Regression" : LogisticRegression(),
         "kNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier(),
         "SVC": SVC()}

#creat a function to fit and score models
def fit_and_score(models, x_train,x_test,y_train,y_test):
    """
    Fits and evaluate given machine learning models.
    models : a dict of different scikit_learn machine learning models
    x_train: train data (no label)
    x_test: test data(no label)
    y_train:training labels
    y_test: testing labels
    """
    #set random seed
    np.random.seed(42)

    #Make a dictionary to keep model score
    model_scores = {}
    # Loop through model
    for name,model in models.items():
        #fit the model to the data
        model.fit(x_train, y_train)
        # Evaluate the model and append its score to model_score
        model_scores[name] = model.score(x_test,y_test)
    return model_scores


# In[35]:


model_scores = fit_and_score(models=models,
                             x_train=x_train,
                             x_test=x_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores


# # Model comparison

# In[36]:


model_compare = pd.DataFrame(model_scores,index=["accuracy"])
model_compare.T.plot.bar();


# # Now we've got a baseline model... and we know a model's first predictions aren't always what we should based our next steps off.
# what should do?
# 
# Let's look at the following:
# * Hyperparameter tuning 
# * feature importance
# * confusion matrix
# * cross-validation
# * precision
# * Recall
# * F1 score
# * classification report
# * ROC curve
# * Area under the curve(AUC)

# # Hyperparameter tuning

# In[37]:


# Let's tune KNN

train_scores = []
test_scores = []

# create a list of differnt values for n_neighbors
neighbors = range(1,21)

# setup = KNN instance
Knn = KNeighborsClassifier()

# loop through different n neighbors
for i in neighbors:
    Knn.set_params(n_neighbors=i)
    
    # fit the algorithms
    Knn.fit(x_train,y_train)
    
    #update the training score list
    train_scores.append(Knn.score(x_train,y_train))
    
    # update the testing scores list
    test_scores.append(Knn.score(x_test,y_test))


# In[38]:


test_scores


# In[39]:


train_scores


# In[40]:


plt.plot(neighbors,train_scores,label="Train score")
plt.plot(neighbors,test_scores,label="Test score")
plt.xlabel("Number of neighbore")
plt.ylabel("Model score")
plt.legend()


print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# # Hyperparameter tuning with the RandomizedSearchCV 
# we're going to tune:
# 
# * LogisticRegression 
# * RandomForestClassifier

# In[41]:


## create a hyperparameter grid For LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# create a hyperparameter grid for RandomForestclassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth":[None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# # Now we've tune LogesticRegression()

# In[42]:


# Tune LogisticRegression
np.random.seed(42)

#setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                               param_distributions=log_reg_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)
# Fit random hyperparameters search model for LogisticRegression
rs_log_reg.fit(x_train, y_train)


# In[43]:


rs_log_reg.best_params_


# In[44]:


rs_log_reg.score(x_test,y_test)


# # Now we've tuned LogesticRegression(), let's do the same for Randomforestclassifier()......

# In[45]:


# setup random seed
np.random .seed(42)

# setup random forest hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                          param_distributions=rf_grid,
                          cv=5,
                          n_iter=20,
                          verbose=True)

#Fit random hyperparameter search model foe randimForestclassifier()
rs_rf.fit(x_train,y_train)


# In[46]:


# Find the the best hyperparameters
rs_rf.best_params_


# In[47]:


# Evaluate the randomized search RandomForestClassifier model
rs_rf.score(x_test,y_test)


# # Hyperparameter tuning with the GridSearchCV
#     
# since RandomForestClassifier model provides the best scores so far,
# we'll try and improve them again using GridsearchCV

# In[49]:


# create a hyperparameter grid for RandomForestclassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth":[None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}

# setup random seed
np.random .seed(42)

# setup random forest hyperparameter search for RandomForestClassifier
gs_rf= GridSearchCV(RandomForestClassifier(),
                    param_grid=rf_grid,
                    cv=5,
                    n_jobs=20,
                    verbose=True)

#Fit random hyperparameter search model for randomForestclassifier()
gs_rf.fit(x_train,y_train)


# In[50]:


gs_rf.score(x_test,y_test)


# In[51]:


gs_rf.best_params_


# In[ ]:





# # Evaluating our tuned machine learning classifier, beyond accuracy
# 
# * ROC curve and AUC score
# * confusion matrix
# * classfication report
# * precision
# * recall
# * F1-score
# 
# ...and it would be great if cross-validation was used where possible
# 
# 
# To make comparision and evaluate our trained model first we need to make predictions.
# 

# In[67]:


## Make predictions with tuned models
y_preds = rs_rf.predict(x_test)


# In[68]:


y_preds


# In[69]:


y_test


# # Plot ROC curve and calculate and calculate AUC metrics
# * ROC curve and AUC matrics of Grid searchCV of RandonForestClassifier

# In[73]:


plot_roc_curve(gs_rf,x_test,y_test)


# # Plot ROC curve and calculate and calculate AUC metrics¶
# * ROC curve and AUC matrics of RandomizedSearchCV  of RandomForestClassifier

# In[74]:


plot_roc_curve(rs_rf,x_test,y_test)


# In[75]:


## confusion matrix
print(confusion_matrix(y_test,y_preds))


# In[76]:


sns.set(font_scale=1.5)

def plot_conf_mat(y_test,y_preds):
    
    """
    plot a nice looking confusion matrix using seaborn's heatmap()
    """
    
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test,y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("True label")
    plt.ylabel("predicted label")
    
   
    
plot_conf_mat(y_test,y_preds)


# # Now lets get a classification report as well as cross-validation precision and f1-score

# In[77]:


print(classification_report(y_test, y_preds))


# # Building prediction system for GridsearchCV and RandomizedSearchCV  of RandomForestClassifier model

# In[78]:


gs_rf.predict([[52,1,0,125,212,0,1,168,0,1.0,2,2,3]])


# In[79]:


rs_rf.predict([[52,1,0,125,212,0,1,168,0,1.0,2,2,3]])


# In[ ]:





# # 6. Experimentation
# 
# If you haven't hit your evaluation metric yet...ask yourself....
# 
# * could you collect more data?
# * could you try a better model? Like CatBoost or XGBoost?
# * could you improve the current models? (beyond what we've done so far)
# * If your model is good enough (you have hit your evaluation metric) how would you export it and share it with other?
#       

# In[ ]:




