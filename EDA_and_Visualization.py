#!/usr/bin/env python
# coding: utf-8

# Importing the libraries

# In[34]:


import pandas as pd
import plotly as px
import matplotlib as plt
import seaborn as sns

import plotly.express as epx

import plotly.graph_objs as go
from ipywidgets import interact




# ## Exploratory Data Analysis and Visualization - Coursera Dataset

# Loading our dataset, and specific sheet selecting from excel 

# In[12]:


df_coursera = pd.read_excel('df_edu.xlsx', sheet_name='Cleaned Data Source 1')


# Viewing the data

# In[16]:


df_coursera.head()


# Getting Column Names

# In[40]:


df_coursera.keys()


# Getting the information

# In[17]:


df_coursera.info()


# Describing the data

# In[18]:


df_coursera.describe()


# Finding the shape

# In[32]:


df_coursera.shape


# Finding the unique values in the columns

# In[20]:


df_coursera['Learning Product Type'].unique()


# In[21]:


df_coursera['Course Provided By'].unique()


# In[22]:


df_coursera['Course Difficulty'].unique()


# Check for NULL values in dataframe

# In[26]:


df_coursera.isnull().sum()


# Check the datatypes 

# In[28]:


df_coursera.dtypes


# Data filtering 

# In[30]:


df_coursera[df_coursera['Course Difficulty']=='Beginner'].head()


# Correlations

# In[43]:


df_coursera.corr()


# 

# Plotting values in descending order 

# In[37]:


df_coursera['Course Difficulty'].value_counts().plot(kind='bar')


# In[38]:


df_coursera['Learning Product Type'].value_counts().plot(kind='bar')


# Box Plot Visualizations

# In[41]:


df_coursera[['Course Rating']].boxplot()


# In[44]:


df_coursera[['Course Rated By']].boxplot()


# Correlation Heat Map

# In[46]:


sns.heatmap(df_coursera.corr())


# In[ ]:





# ## Exploratory Data Analysis and Visualization - Udemy Dataset
# 

# In[1]:



# In[2]:


df_udemy = pd.read_excel('df_edu.xlsx', sheet_name='Data set Source 3')


# In[3]:


df_udemy.head()


# In[4]:


df_udemy.keys()


# In[5]:


df_udemy.info()


# In[6]:


df_udemy.describe()


# In[7]:


df_udemy.shape


# In[8]:


df_udemy['isPaid'].unique()


# In[9]:


df_udemy['category'].unique()


# In[10]:


df_udemy['instructionalLevel'].unique()


# In[11]:


df_udemy.isnull().sum()


# In[12]:


df_udemy.dtypes


# In[13]:


df_udemy[df_udemy['instructionalLevel']=='Expert Level'].head()


# In[14]:


df_udemy.corr()


# In[35]:


#Interactive Bar chart of number of courses in each category


# Run the code for the interactive visualization to show

# In[16]:


fig = epx.bar(df_udemy, x='category', color='category', hover_data=['numSubscribers', 'numReviews'])
fig.show()


# In[17]:


#Scatter plot of price vs number of subscribers


# Run the code for the interactive visualization to show

# In[18]:


fig = epx.scatter(df_udemy, x='price', y='numSubscribers', color='category', hover_data=['title'])
fig.show()


# In[19]:


#Sunburst chart of the instructional level and category


# Run the code for the interactive visualization to show

# In[20]:


fig = epx.sunburst(df_udemy, path=['instructionalLevel', 'category'], values='numSubscribers',hover_data=['title'])
fig.show()


# In[21]:


# Box plot of the number of reviews by instructional level


# Run the code for the interactive visualization to show
# 

# In[22]:


fig = epx.box(df_udemy, x='instructionalLevel', y='numReviews', color='instructionalLevel', hover_data=['title'])
fig.show()


# ## Exploratory Data Analysis and Visualization - Countries Dataset

# In[23]:


df_countries = pd.read_excel('df_edu.xlsx', sheet_name='Data set Source 2')


# In[24]:


df_countries.head()


# In[25]:


df_countries.keys()


# In[26]:


df_countries.info()


# In[27]:


df_countries.shape


# In[28]:


df_countries.describe()


# In[29]:


df_countries.isnull().sum()


# In[30]:


df_countries['Month'].unique()


# In[31]:


df_countries.corr


# In[32]:


df_countries.dtypes


# Run the code for the interactive visualization to show

# In[33]:


x = df_countries['Month']
y = df_countries.drop('Month', axis=1)
@interact(countries=list(y.columns))
def plot_line_chart(countries):
    trace = go.Scatter(x=x, y=y[countries], mode='lines+markers')
    layout = go.Layout(title=countries + ' Search Interest Over Time', xaxis=dict(title='Month'), yaxis=dict(title='Search Interest'))
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


# In[ ]:





# In[ ]:





# In[ ]:




