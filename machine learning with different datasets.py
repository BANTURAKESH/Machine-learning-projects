#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Topics Iam covering in this notebook.
# 
# * Basics of Data Mining: dimensionality reduction
# * Principal Component Analysis (PCA)
# * Linear Discriminant Analysis (LDA)
# * Regression analysis
# * k-Nearest Neighbors algorithm (K-NN)
# * Random forest
# * Linear regression 
# * Support Vector Machine (SVM)
# * Decision trees
# * Heat-Maps
# * Hierarchical clustering
# * Training and Testing The Data
# * label Encoding
# * Naïve Bayes Classifier Algorithm
# * K-Means Clustering(Elbow Plot Method)
# * K-Folds cross-validation
# * Building confusion_matrix

# In[ ]:





# # Importing the dependencies

# In[1]:


# To filter the warnings.
import warnings
warnings.filterwarnings("ignore")

import pandas as pd # To Read the datasets.
import numpy as np

# For Data visualization.
# The Finding of trends and correlations in our data by representing it pictorially is called Data visualizations.
import matplotlib.pyplot as plt
import seaborn as sns

# To Load the datasets from sklearn library.
from sklearn import datasets

from sklearn.metrics import confusion_matrix # To Build confusion matrix.
from sklearn.metrics import classification_report # To Get the precision , recall , f1-scores.
from sklearn.preprocessing import LabelEncoder # converting the labels into a numeric form. 
from sklearn.model_selection import cross_val_score # To Evaluate the  Best Model. 

#To calculate the mean intra-cluster distance & the mean nearest-cluster distance. 
from sklearn.metrics import silhouette_score 
import scipy.cluster .hierarchy as sch # To construct Dendrograms

# Standardize features by removing the mean and scaling to unit variance.
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler #Transform features by scaling each feature to a given range.

# To split the data into training and testing data
from sklearn.model_selection import train_test_split



# Different types of classifiers I used.
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[ ]:





# In[ ]:





# # Principal Component Analysis (PCA)
# * The Principal Component Analysis is a popular unsupervised learning technique for reducing the dimensionality of data.
# * It increases interpretability yet, at the same time, it minimizes information loss.
# * It helps to find the most significant features in a dataset and makes the data easy for plotting in 2D and 3D.
# 

# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# In[4]:


cancer.keys()


# In[5]:


print(cancer['DESCR'])


# In[6]:


# Creating a Data Frame
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()


# In[7]:


df.describe()


# In[8]:


x=df
y= cancer.target


# In[224]:


x.head()


# In[ ]:





# # Data Standardization

# In[226]:


# Scaling the data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[227]:


scaler = StandardScaler()
scaler.fit(x)


# In[284]:


scaled_data = scaler.transform(x)


# In[285]:


scaled_data


# In[266]:


x_train,x_test,y_train,y_test = train_test_split(scaled_data,y,test_size=0.2,random_state=2)


# In[267]:


from sklearn.linear_model import LogisticRegression

model = LinearRegression()
model.fit(x_train,y_train)

model.score(x_test,y_test)


# In[268]:


from sklearn.decomposition import PCA


# # Reducing 30 Dimenctions to 2 Dimenction
# 
# * Reducing the dimensionality of data.

# In[269]:


pca=PCA(n_components=2)


# In[270]:


pca.fit(x)


# In[271]:


x_pca = pca.transform(x)


# In[272]:


x.shape


# In[273]:


x_pca.shape


# In[274]:


scaled_data


# In[275]:


x_pca


# In[276]:


pca.explained_variance_ratio_


# In[277]:


pca.n_components_


# In[296]:


x_train,x_test,y_train,y_test = train_test_split(x_pca,y,test_size=0.2,random_state=2)


# # LinearDiscriminantAnalysis

# In[297]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[298]:


model = LinearDiscriminantAnalysis()
model.fit(x_train,y_train)


# In[305]:


model.score(x_test,y_test)


# In[303]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel("First principle component ")
plt.ylabel("second principle component")


# In[ ]:





# In[ ]:





# In[ ]:





# # LinearRegression
# * LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the datasets.
# * the targets predicted by the linear approximation.
# * Linear Regression Linear Regression is a machine learning algorithm based on supervised learning.
# * It performs a regression task. Regression models a target prediction value based on independent variables.
# * It is mostly used for finding out the relationship between variables and forecasting

# # Training and Testing The Data(LinearRegression)
# How to use train_test_split Method using LinearRegressionClassifier

# In[5]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[28]:


df= pd.read_csv("carprices.csv")
df.head(5)


# # Data preprocessing separating the futures and targets

# In[31]:


x = df.drop(columns=['Sell Price($)'],axis=1)
y = df['Sell Price($)']


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# In[12]:


print(x.shape, x_train.shape, x_test.shape)


# In[13]:


len(x_train)


# In[14]:


len(x_test)


# In[15]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()


# In[16]:


clf.fit(x_train,y_train)


# In[17]:


clf.predict(x_test) 


# In[19]:


clf.score(x_test,y_test)


# In[ ]:





# In[ ]:





# #  Spport Vector Machine
# * The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.
# * SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine
# 
# ![image.png](attachment:image.png)
# 

# In[174]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[175]:


dir(iris)


# In[176]:


iris.feature_names


# In[177]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[178]:


df['target'] = iris.target
df.head()


# In[179]:


iris.target_names


# In[180]:


df[df.target==0].head()


# In[181]:


df['flower_name'] = df.target.apply(lambda x:iris.target_names[x])
df.head()


# In[182]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[183]:


df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2] 


# # sepal length vs sepal width

# In[185]:


plt.xlabel('sepal length (cm)') 
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='.')


# # petal length vs petal width

# In[186]:


plt.xlabel('petal length (cm)') 
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green',marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue',marker='.')


# In[187]:


# splitting the data futures and targets


# In[188]:


x = df.drop(columns=['target','flower_name'], axis=1)
y = df['target']


# In[191]:


from sklearn.model_selection import train_test_split


# In[192]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=2)


# In[193]:


print(x.shape, x_train.shape, x_test.shape)


# In[194]:


from sklearn.svm import SVC


# In[195]:


model = SVC()


# In[196]:


model.fit(x_train,y_train)


# In[197]:


model.score(x_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # k-nearest neighbors Algorithm 
# * Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. 
# * Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.
# * The -neighbors classification in KNeighborsClassifier is the most commonly used technique.
# * The optimal choice of the value  is highly data-dependent: in general a larger  suppresses the effects of noise, but makes the classification boundaries less distinct.

# In[198]:


len(x_train)


# In[199]:


len(x_test)


# # Creating(k-nearest neighbors Classifier)

# In[200]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(x_train,y_train)


# In[201]:


knn.score(x_test,y_test)


# In[202]:


from sklearn.metrics import confusion_matrix

y_pred = knn. predict(x_test)

cm = confusion_matrix(y_test,y_pred)
cm


# In[206]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel("predicted")
plt.ylabel("Truth")


# # Classification Report

# In[207]:


from sklearn.metrics import classification_report


# In[209]:


print(classification_report(y_pred,y_test))


# # Accuracy = TP+TN/TP+FP+FN+TN

# # Precision = TP/TP+FP

# # Recall = TP/TP+FN

# # F1 Score = 2*(Recall * Precision) / (Recall + Precision)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Decision Tree Algorithm
# * Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression.
# * The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. 
# * A tree can be seen as a piecewise constant approximation.

# In[44]:


from sklearn.preprocessing import LabelEncoder
from sklearn import tree


# In[11]:


#df = pd.read_csv("salaries.csv")


# # label Encoding
# converting the labels into numeric form

# In[46]:


# loading the label encoding function 


# In[47]:


label_encoder = LabelEncoder()


# In[48]:


labels = label_encoder.fit_transform(df.company)
df['company_n']=labels 


# In[49]:


labels = label_encoder.fit_transform(df.job)
df['job_n']=labels 


# In[50]:


labels = label_encoder.fit_transform(df.company)
df['degree_n']=labels 


# In[52]:


df.head()


# In[53]:


#splitting the data into x and y
x = df.drop(columns=['salary_more_then_100k'])
y = df['salary_more_then_100k']


# In[55]:


x = df.drop(columns=['company','job','degree','salary_more_then_100k'])


# In[58]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[61]:


clf = tree.DecisionTreeClassifier()


# In[62]:


clf.fit(x_train,y_train)


# In[63]:


clf.score(x_test,y_test)


# In[64]:


clf.predict([[2,0,2]])


# # RandomForest Algorithm
# * A random forest classifier.
# 
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.
# 
# 

# In[65]:


# Loading digits dataset from sklearn
from sklearn.datasets import load_digits
digits= load_digits()


# In[66]:


dir(digits)


# In[67]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])


# In[68]:


digits.data[:5]


# In[69]:


df = pd.DataFrame(digits.data)
df.head()


# # Adding the target

# In[70]:


df['target']=digits.target
df.head()


# In[71]:


# splitting x and y futures and targets
x = df.drop(columns=['target'], axis=1)
y = df['target']


# In[ ]:





# In[ ]:





# In[74]:


x.shape,y.shape


# # Splitting the data into Training  and Testing data

# In[75]:


from sklearn.model_selection import train_test_split 


# In[76]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[ ]:





# In[ ]:





# # Model Training 

# In[79]:


from sklearn.ensemble import RandomForestClassifier


# In[80]:


model = RandomForestClassifier()


# In[81]:


model.fit(x_train,y_train)


# In[82]:


model.score(x_test,y_test)


# In[83]:


y_predicted = model.predict(x_test)


# In[84]:


from sklearn.metrics import confusion_matrix


# # Building confusion_matrix

# In[85]:


cm = confusion_matrix(y_test,y_predicted)
cm


# In[86]:


import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')


# In[ ]:





# In[ ]:





# # There are different ways to perform Training step
# 
# option 1
# 
# * use all available data for training and test on same dataset
# 
# option 2
# 
# * Split available dataset into training and test sets [train_test_split Method]
# *The train_test_split() method is used to split our data into train and test sets. 
# 
# *First, we need to divide our data into features (X) and labels (y). The dataframe gets divided into X_train,X_test , y_train and y_test. X_train and y_train sets are used for training and fitting the model. The X_test and y_test sets are used for testing the model if it’s predicting the right outputs/labels. we can explicitly test the size of the train and test sets. It is suggested to keep our train sets larger than the test sets.
# 
# 
# 
# option 3
# 
# * K fold Cross Validation
# 
# 

# # K-Folds cross-validation
# 
# * Many times we will think of which machine learning model should we use for a given problem
# * kFold-Cross Validation allows us to evaluate performance of a model by creating K Fold of given dataset.
# * This is better then doing train_test_split.
# 
# we'll Look
# * Cross Validation
# * kfold
# * cross_val_score function 

# # By using above digits dataset we'll perform K-Fold Cross validation
#  
# BY using 3 models:-
# * LinearRegression
# * Support Vector machine
# * RandomForestClssifier

# In[87]:


import warnings
warnings.filterwarnings('ignore')


# In[88]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[89]:


model = LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[90]:


model = SVC()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[91]:


model = RandomForestClassifier()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# # Know using K-Fold cross validation

# In[92]:


from sklearn.model_selection import cross_val_score


# In[93]:


cross_val_score(LogisticRegression(), digits.data, digits.target)


# In[94]:


cross_val_score(SVC(), digits.data, digits.target)


# In[95]:


cross_val_score(RandomForestClassifier(n_estimators=50), digits.data, digits.target)


# In[ ]:





# # K-Means Clustering.

# * K-means is an unsupervised learning method for clustering data points. The algorithm iteratively divides data points into K clusters by minimizing the variance in each cluster.
# 
# * Here, we will show you how to estimate the best value for K using the elbow method, then use K-means clustering to group the data points into cluste

# In[7]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[8]:


df = pd.read_csv('income.csv')
df.head()


# In[9]:


plt.scatter(df['Age'],df['Income($)'])


# In[10]:


from sklearn.cluster import KMeans


# In[11]:


km = KMeans(n_clusters=3)
km


# In[6]:


y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted


# In[101]:


df['cluster'] = y_predicted
df.head()


# In[102]:


x = df.drop(columns=['cluster'], axis=1)
y = df['cluster']


# In[103]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='blue')
plt.scatter(df3.Age,df3['Income($)'],color='red')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()


# In[104]:


df.shape


# In[105]:


scaler =MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
df


# In[106]:


scaler =MinMaxScaler()
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
df


# In[107]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted


# In[108]:


df['cluster'] = y_predicted
df.head()


# In[109]:


km.cluster_centers_


# In[110]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='blue')
plt.scatter(df3.Age,df3['Income($)'],color='red')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'purple',marker='*',label='centroid')
plt.legend()


# # Elbow Plot Method

# # km.inertia:-
# * Inertia measures how well a dataset was clustered by K-Means.
# * It is calculated by measuring the distance between each data point and its centroid, squaring this distance, and summing these squares across one cluster.
# * A good model is one with low inertia AND a low number of clusters ( K )

# In[ ]:


# The SSE is defined as the sum of the squared Euclidean distances of each point to its closest centroid.
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)


# In[112]:


sse


# In[113]:


plt.xlabel('k')
plt.ylabel('sum of square error')
plt.plot(k_rng,sse)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Hierarchical clustering
# * Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging or splitting them successively.
# 
# * This hierarchy of clusters is represented as a tree (or dendrogram).
# 
# * The root of the tree is the unique cluster that gathers all the samples, the leaves being the clusters with only one sample

# In[114]:


from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster .hierarchy as sch


# In[115]:


datasubset = df.loc[:,["Age","Income($)"]]


# In[116]:


plt.figure(figsize=(10,7))
plt.scatter(datasubset[["Age"]], datasubset[["Income($)"]],s=100,c='green')


# In[117]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))

dendrogram = sch.dendrogram(sch.linkage(datasubset,method = "ward"))
plt.title("Dendogram")
plt.xlabel("Name")
plt.ylabel("Euclidean distance")
plt.show()
# check for largest distance vertically with crossing any horizontal line


# In[123]:


cluster = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")
cluster.fit_predict(datasubset)


# In[124]:


cl = cluster.fit_predict(datasubset)


# # silhouette score
# 
# * The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample.

# In[125]:


from sklearn.metrics import silhouette_score


# In[126]:


silhouette_score(datasubset,cl)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Naïve Bayes Classifier Algorithm
# * supervised learning algorithm
# * It is mainly used in text classification that includes a high-dimensional training dataset.
# * It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.
# * Some popular examples are spam filtration, Sentimental analysis, and classifying articles.
# ![naive-bayes-classifier-algorithm.png](attachment:naive-bayes-classifier-algorithm.png)
#  

# In[874]:


df = pd.read_csv('titanic.csv')
df.head()


# In[875]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()


# In[876]:


target =df.Survived
inputs = df.drop('Survived',axis='columns')


# In[877]:


dumies = pd.get_dummies(inputs.Sex)
dumies.head(3)


# In[878]:


inputs = pd.concat([inputs,dumies],axis='columns')
inputs.head(3)


# In[879]:


inputs.drop('Sex',axis='columns',inplace=True)
inputs.head(3)


# # Finding Null Values

# In[880]:


inputs.columns[inputs.isnull().any()]


# In[881]:


inputs.Age[:10]


# # we'll Fill NaN values

# In[882]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head(10)


# In[883]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test  = train_test_split(inputs,target,test_size=0.3)


# In[884]:


len(x_train)


# In[885]:


len(x_test)


# In[886]:


len(inputs)


# In[887]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[888]:


model.fit(x_train,y_train)


# In[889]:


model.score(x_test,y_test)


# In[890]:


x_test


# In[891]:


y_test[:10]


# In[892]:


model.predict(x_test[:10])


# In[893]:


model.predict_proba(x_test[:10])


# In[ ]:





# #  What is label Encoding?
# # converting the labels into numeric form

# In[306]:


from sklearn.preprocessing import LabelEncoder


# In[307]:


cancer = pd.read_csv('cancer.csv')


# In[308]:


cancer.head()


# # Finding the count of different labels

# In[310]:


cancer['diagnosis'].value_counts()


# # load the label encoder function

# In[312]:


label_encoder = LabelEncoder()


# In[313]:


labels = label_encoder.fit_transform(cancer.diagnosis)


# # Appending the Label to the DataFrame

# In[315]:


cancer['target']=labels


# In[316]:


cancer.head()


# # 0--> Benine

# # 1--> Malignant

# In[318]:


cancer['diagnosis'].value_counts()


# In[ ]:




