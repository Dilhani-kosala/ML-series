#!/usr/bin/env python
# coding: utf-8

# # Addition of two numbers using ML

# In[1]:


import pandas as pd


# ## importing data set

# In[2]:


data=pd.read_csv('mlcsv')


# ## show the data set

# In[3]:


data


# ## display the head of data set

# In[4]:


data.head()


# ## display the bottom data set

# In[5]:


data.tail()


# ## get number of rows & columns

# In[6]:


data.shape


# ## Preprocessing - find there is null values or not

# In[7]:


data.info()


# ## Do the EDA - import matplot library

# In[8]:


import matplotlib.pyplot as plt


# ## relationship between dependent & independent variable

# In[9]:


plt.scatter(data['x'],data['sum'])


# In[10]:


plt.scatter(data['y'],data['sum'])


# ## store features (inputs) in Matrix (X) & store response(target) in Vectory (Y)

# In[11]:


X= data[['x','y']]
y= data ['sum']


# ## train / test split using sklearn library

# ### split the data into two parts (training set & testing set)

# In[12]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
 X,y, test_size=0.33, random_state=42)


# ## get inputs (x,y)

# In[13]:


X_train


# ## import & train the model

# In[14]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)


# ## model's prediction performance using score()

# In[15]:


model.score(X_train,y_train)


# In[16]:


model.score(X_test,y_test)


# ## compare the results using prediction

# In[17]:


y_pred=model.predict(X_test)
y_pred


# ## compare the results using data frame 

# In[18]:


df=pd.DataFrame({'Actual':y_test,'prediction':y_pred})
df


# ## prediction on new samples (input any 2 values and get the output)

# In[19]:


model.predict([[100,52]])


# ## save the model using joblib or pickel

# In[20]:


import joblib
joblib.dump(model,'model_joblib')


# ### load the model

# In[21]:


model=joblib.load('model_joblib')


# ### prediction (get output using any 2 inputs)

# In[22]:


model.predict([[25,75]])


# ## training for the entire data set

# In[23]:


X = data [['x','y']]
y = data['sum']
model=LinearRegression()
model.fit(X,y)


# ## again save the whole model

# In[24]:


import joblib
joblib.dump(model,'model_joblib')
model=joblib.load('model_joblib')
model.predict([[50,20]])


# ## GUI (use tkinter,applying lables)

# In[46]:


def show_entry_fields():
 p1=float(e1.get())
 p2=float(e2.get())
 model=joblib.load('model_joblib')
 result=model.predict([[p1,p2]])
 Label(master,text='sum is=').grid(row=4)
 Label(master,text=result).grid (row=5)
 print("Sum is",result)

from tkinter import *
import joblib

master=Tk()
master.title("Addition of two numbers using ML")
label=Label(master,text="addition of two numbers using ML",bg='purple',fg='white').grid(row=0,columnspan=2)

Label(master,text="Enter first number").grid(row=1)
Label(master,text="Enter second number").grid(row=2)
e1=Entry(master)
e2=Entry(master)
e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
Button(master,text='predict',command=show_entry_fields).grid()
mainloop()


# In[ ]:




