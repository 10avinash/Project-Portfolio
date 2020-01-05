#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn


# In[3]:


import matplotlib.pyplot as plt
import pandas as pd


# In[7]:


xlist = pd.read_excel('xlist.xlsx')
ylist = pd.read_excel('ylist.xlsx')

import numpy as np
xlist = xlist.values  #converting to numpy for flexibility
ylist = ylist.values  #converting to numpy for flexibility

te = pd.read_csv('TransE.csv')
strow = list(te)
te.loc[-1] = strow  # adding a row
te.index = te.index + 1  # shifting index
te = te.sort_index()  # sorting by index
te = te.iloc[:,1]

#xlist.shape
xtensor = torch.zeros(5000,400,dtype=torch.float)
ytensor = torch.zeros(5000,400,dtype=torch.float)
tetensor = torch.zeros(5000,1,dtype=torch.float)
xmax = torch.zeros(5000,1,dtype = torch.float)
ymax = torch.zeros(5000,1,dtype = torch.float)
te = te.values
te= list(map(float, te))


# In[8]:


for i in range(5000):
  xtensor[i] = torch.tensor(xlist[i])
  ytensor[i] = torch.tensor(ylist[i])
  tetensor[i] = torch.tensor(te[i])  


# In[9]:


xmax = torch.max(xtensor)
ymax = torch.max(ytensor)
xtensor = torch.div(xtensor,xmax)
ytensor = torch.div(ytensor,ymax)


# In[10]:


tepred = tetensor
lag = 1
ztensor = torch.cat((xtensor,ytensor),1)
ztensor.size()


# In[12]:


lagtensor = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor[i] = torch.FloatTensor([1.0,1.0,1.0])
#lagtensor
output = torch.cat((tetensor,lagtensor),1)
sampleztensor = torch.ones(1,800)
output


# In[13]:


xlist2 = pd.read_excel('xl2.xlsx')
ylist2 = pd.read_excel('yl2.xlsx')
print(xlist2.shape)
xlist2 = xlist2.iloc[:,200:600]
print(xlist2.shape)


# In[14]:


import numpy as np
xlist2 = xlist2.values  #converting to numpy for flexibility
ylist2 = ylist2.values  #converting to numpy for flexibility

te2 = pd.read_csv('TE222.csv')
strow2 = list(te2)
te2.loc[-1] = strow2  # adding a row
te2.index = te2.index + 1  # shifting index
te2 = te2.sort_index()  # sorting by index
te2 = te2.iloc[:,1]
te2 = te2.values
te2= list(map(float, te2))

xtensor2 = torch.zeros(5000,400,dtype=torch.float)
ytensor2 = torch.zeros(5000,400,dtype=torch.float)
tetensor2 = torch.zeros(5000,1,dtype=torch.float)
xmax2 = torch.zeros(5000,1,dtype = torch.float)
ymax2 = torch.zeros(5000,1,dtype = torch.float)

temax2 = torch.zeros(1,dtype = torch.float)
temax2 = torch.max(tetensor2)
tepred2 = torch.zeros(5000,1,dtype=torch.float)
tepred2 = torch.div(tetensor2,temax2)
ztensor2 = torch.cat((xtensor2,ytensor2),0)
ztensor2.size()


# In[15]:


for i in range(5000):
  xtensor2[i] = torch.tensor(xlist2[i])
  ytensor2[i] = torch.tensor(ylist2[i])
  tetensor2[i] = torch.tensor(te2[i])  


# In[16]:


xmax2 = torch.max(xtensor2)
ymax2 = torch.max(ytensor2)
xtensor2 = torch.div(xtensor2,xmax2)
ytensor2 = torch.div(ytensor2,ymax2)
ztensor2 = torch.cat((xtensor2,ytensor2),1)
ztensor2.size()

lagtensor2 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor2[i] = torch.FloatTensor([2.0,2.0,2.0])
output2 = torch.cat((tetensor2,lagtensor2),1)

netoutput = torch.cat((output,output2),0)
netinput = torch.cat((ztensor,ztensor2),0)


# In[17]:


xlist3 = pd.read_excel('2xl2.xlsx')
ylist3 = pd.read_excel('2yl3.xlsx')
xlist3 = xlist3.iloc[:,200:600]
xlist3 = xlist3.values  #converting to numpy for flexibility
ylist3 = ylist3.values  #converting to numpy for flexibility
te3 = pd.read_csv('TE232.csv')
strow3 = list(te3)
te3.loc[-1] = strow3  # adding a row
te3.index = te3.index + 1  # shifting index
te3 = te3.sort_index()  # sorting by index
te3 = te3.iloc[:,1]
te3 = te3.values
te3 = list(map(float, te3))

xtensor3 = torch.zeros(5000,400,dtype=torch.float)
ytensor3 = torch.zeros(5000,400,dtype=torch.float)
tetensor3 = torch.zeros(5000,1,dtype=torch.float)
xmax3 = torch.zeros(5000,1,dtype = torch.float)
ymax3 = torch.zeros(5000,1,dtype = torch.float)

temax3 = torch.zeros(1,dtype = torch.float)
temax3 = torch.max(tetensor3)
tepred3 = torch.zeros(5000,1,dtype=torch.float)
tepred3 = torch.div(tetensor3,temax3)
ztensor3 = torch.cat((xtensor3,ytensor3),0)
ztensor3.size()

for i in range(5000):
  xtensor3[i] = torch.tensor(xlist3[i])
  ytensor3[i] = torch.tensor(ylist3[i])
  tetensor3[i] = torch.tensor(te3[i])  
xmax3 = torch.max(xtensor3)
ymax3 = torch.max(ytensor3)
xtensor3 = torch.div(xtensor3,xmax3)
ytensor3 = torch.div(ytensor3,ymax3)
ztensor3 = torch.cat((xtensor3,ytensor3),1)
ztensor3.size()

lagtensor3 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor3[i] = torch.FloatTensor([2.0,3.0,2.0])
output3 = torch.cat((tetensor3,lagtensor3),1)

netoutput = torch.cat((netoutput,output3),0)
netinput = torch.cat((netinput,ztensor3),0)


# In[18]:


xlist4 = pd.read_excel('3xl3.xlsx')
ylist4 = pd.read_excel('3yl3.xlsx')
xlist4 = xlist4.iloc[:,200:600]
xlist4 = xlist4.values  #converting to numpy for flexibility
ylist4 = ylist4.values  #converting to numpy for flexibility
te4 = pd.read_csv('TE332.csv')
strow4 = list(te4)
te4.loc[-1] = strow4  # adding a row
te4.index = te4.index + 1  # shifting index
te4 = te4.sort_index()  # sorting by index
te4 = te4.iloc[:,1]
te4 = te4.values
te4 = list(map(float, te4))

xtensor4 = torch.zeros(5000,400,dtype=torch.float)
ytensor4 = torch.zeros(5000,400,dtype=torch.float)
tetensor4 = torch.zeros(5000,1,dtype=torch.float)
xmax4 = torch.zeros(5000,1,dtype = torch.float)
ymax4 = torch.zeros(5000,1,dtype = torch.float)

temax4 = torch.zeros(1,dtype = torch.float)
temax4 = torch.max(tetensor4)
tepred4 = torch.zeros(5000,1,dtype=torch.float)
tepred4 = torch.div(tetensor4,temax4)
ztensor4 = torch.cat((xtensor4,ytensor4),0)
ztensor4.size()

for i in range(5000):
  xtensor4[i] = torch.tensor(xlist4[i])
  ytensor4[i] = torch.tensor(ylist4[i])
  tetensor4[i] = torch.tensor(te4[i])  
xmax4 = torch.max(xtensor4)
ymax4 = torch.max(ytensor4)
xtensor4 = torch.div(xtensor4,xmax4)
ytensor4 = torch.div(ytensor4,ymax4)
ztensor4 = torch.cat((xtensor4,ytensor4),1)
ztensor4.size()

lagtensor4 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor4[i] = torch.FloatTensor([3.0,3.0,2.0])
output4 = torch.cat((tetensor4,lagtensor4),1)

netoutput = torch.cat((netoutput,output4),0)
netinput = torch.cat((netinput,ztensor4),0)


# In[19]:


xlist5 = pd.read_excel('4xl3.xlsx')
ylist5 = pd.read_excel('4yl3.xlsx')
xlist5 = xlist5.iloc[:,200:600]
xlist5 = xlist5.values  #converting to numpy for flexibility
ylist5 = ylist5.values  #converting to numpy for flexibility
te5 = pd.read_csv('TE333.csv')
strow5 = list(te5)
te5.loc[-1] = strow5  # adding a row
te5.index = te5.index + 1  # shifting index
te5 = te5.sort_index()  # sorting by index
te5 = te5.iloc[:,1]
te5 = te5.values
te5 = list(map(float, te5))

xtensor5 = torch.zeros(5000,400,dtype=torch.float)
ytensor5 = torch.zeros(5000,400,dtype=torch.float)
tetensor5 = torch.zeros(5000,1,dtype=torch.float)
xmax5 = torch.zeros(5000,1,dtype = torch.float)
ymax5 = torch.zeros(5000,1,dtype = torch.float)

temax5 = torch.zeros(1,dtype = torch.float)
temax5 = torch.max(tetensor5)
tepred5 = torch.zeros(5000,1,dtype=torch.float)
tepred5 = torch.div(tetensor5,temax5)
ztensor5 = torch.cat((xtensor5,ytensor5),0)
ztensor5.size()

for i in range(5000):
  xtensor5[i] = torch.tensor(xlist5[i])
  ytensor5[i] = torch.tensor(ylist5[i])
  tetensor5[i] = torch.tensor(te5[i])  
xmax5 = torch.max(xtensor5)
ymax5 = torch.max(ytensor5)
xtensor5 = torch.div(xtensor5,xmax5)
ytensor5 = torch.div(ytensor5,ymax5)
ztensor5 = torch.cat((xtensor5,ytensor5),1)
ztensor5.size()

lagtensor5 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor5[i] = torch.FloatTensor([3.0,3.0,3.0])
output5 = torch.cat((tetensor5,lagtensor5),1)

netoutput = torch.cat((netoutput,output5),0)
netinput = torch.cat((netinput,ztensor5),0)


# In[20]:


xlist6 = pd.read_excel('5xl4.xlsx')
ylist6 = pd.read_excel('5yl3.xlsx')
xlist6 = xlist6.iloc[:,200:600]
xlist6 = xlist6.values  #converting to numpy for flexibility
ylist6 = ylist6.values  #converting to numpy for flexibility
te6 = pd.read_csv('TE433.csv')
strow6 = list(te6)
te6.loc[-1] = strow6  # adding a row
te6.index = te6.index + 1  # shifting index
te6 = te6.sort_index()  # sorting by index
te6 = te6.iloc[:,1]
te6 = te6.values
te6 = list(map(float, te6))

xtensor6 = torch.zeros(5000,400,dtype=torch.float)
ytensor6 = torch.zeros(5000,400,dtype=torch.float)
tetensor6 = torch.zeros(5000,1,dtype=torch.float)
xmax6 = torch.zeros(5000,1,dtype = torch.float)
ymax6 = torch.zeros(5000,1,dtype = torch.float)

temax6 = torch.zeros(1,dtype = torch.float)
temax6 = torch.max(tetensor6)
tepred6 = torch.zeros(5000,1,dtype=torch.float)
tepred6 = torch.div(tetensor6,temax6)
ztensor6 = torch.cat((xtensor6,ytensor6),0)
ztensor6.size()

for i in range(5000):
  xtensor6[i] = torch.tensor(xlist6[i])
  ytensor6[i] = torch.tensor(ylist6[i])
  tetensor6[i] = torch.tensor(te6[i])  
xmax6 = torch.max(xtensor6)
ymax6 = torch.max(ytensor6)
xtensor6 = torch.div(xtensor6,xmax6)
ytensor6 = torch.div(ytensor6,ymax6)
ztensor6 = torch.cat((xtensor6,ytensor6),1)
ztensor6.size()

lagtensor6 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor6[i] = torch.FloatTensor([4.0,3.0,3.0])
output6 = torch.cat((tetensor6,lagtensor6),1)

netoutput = torch.cat((netoutput,output6),0)
netinput = torch.cat((netinput,ztensor6),0)


# In[21]:


xlist7 = pd.read_excel('6xl4.xlsx')
ylist7 = pd.read_excel('6yl4.xlsx')
xlist7 = xlist7.iloc[:,200:600]
xlist7 = xlist7.values  #converting to numpy for flexibility
ylist7 = ylist7.values  #converting to numpy for flexibility
te7 = pd.read_csv('TE443.csv')
strow7 = list(te7)
te7.loc[-1] = strow7  # adding a row
te7.index = te7.index + 1  # shifting index
te7 = te7.sort_index()  # sorting by index
te7 = te7.iloc[:,1]
te7 = te7.values
te7 = list(map(float, te7))

xtensor7 = torch.zeros(5000,400,dtype=torch.float)
ytensor7 = torch.zeros(5000,400,dtype=torch.float)
tetensor7 = torch.zeros(5000,1,dtype=torch.float)
xmax7 = torch.zeros(5000,1,dtype = torch.float)
ymax7 = torch.zeros(5000,1,dtype = torch.float)

temax7 = torch.zeros(1,dtype = torch.float)
temax7 = torch.max(tetensor7)
tepred7 = torch.zeros(5000,1,dtype=torch.float)
tepred7 = torch.div(tetensor7,temax7)
ztensor7 = torch.cat((xtensor7,ytensor7),0)
ztensor7.size()

for i in range(5000):
  xtensor7[i] = torch.tensor(xlist7[i])
  ytensor7[i] = torch.tensor(ylist7[i])
  tetensor7[i] = torch.tensor(te7[i])  
xmax7 = torch.max(xtensor7)
ymax7 = torch.max(ytensor7)
xtensor7 = torch.div(xtensor7,xmax7)
ytensor7 = torch.div(ytensor7,ymax7)
ztensor7 = torch.cat((xtensor7,ytensor7),1)
ztensor7.size()

lagtensor7 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor7[i] = torch.FloatTensor([4.0,4.0,3.0])
output7 = torch.cat((tetensor7,lagtensor7),1)

netoutput = torch.cat((netoutput,output7),0)
netinput = torch.cat((netinput,ztensor7),0)


# In[22]:


xlist6 = pd.read_excel('5xl4.xlsx')
ylist6 = pd.read_excel('5yl3.xlsx')
xlist6 = xlist6.iloc[:,200:600]
xlist6 = xlist6.values  #converting to numpy for flexibility
ylist6 = ylist6.values  #converting to numpy for flexibility
te6 = pd.read_csv('TE433.csv')
strow6 = list(te6)
te6.loc[-1] = strow6  # adding a row
te6.index = te6.index + 1  # shifting index
te6 = te6.sort_index()  # sorting by index
te6 = te6.iloc[:,1]
te6 = te6.values
te6 = list(map(float, te6))

xtensor6 = torch.zeros(5000,400,dtype=torch.float)
ytensor6 = torch.zeros(5000,400,dtype=torch.float)
tetensor6 = torch.zeros(5000,1,dtype=torch.float)
xmax6 = torch.zeros(5000,1,dtype = torch.float)
ymax6 = torch.zeros(5000,1,dtype = torch.float)

temax6 = torch.zeros(1,dtype = torch.float)
temax6 = torch.max(tetensor6)
tepred6 = torch.zeros(5000,1,dtype=torch.float)
tepred6 = torch.div(tetensor6,temax6)
ztensor6 = torch.cat((xtensor6,ytensor6),0)
ztensor6.size()

for i in range(5000):
  xtensor6[i] = torch.tensor(xlist6[i])
  ytensor6[i] = torch.tensor(ylist6[i])
  tetensor6[i] = torch.tensor(te6[i])  
xmax6 = torch.max(xtensor6)
ymax6 = torch.max(ytensor6)
xtensor6 = torch.div(xtensor6,xmax6)
ytensor6 = torch.div(ytensor6,ymax6)
ztensor6 = torch.cat((xtensor6,ytensor6),1)
ztensor6.size()

lagtensor6 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor6[i] = torch.FloatTensor([4.0,3.0,3.0])
output6 = torch.cat((tetensor6,lagtensor6),1)

netoutput = torch.cat((netoutput,output6),0)
netinput = torch.cat((netinput,ztensor6),0)


# In[23]:


xlist7 = pd.read_excel('6xl4.xlsx')
ylist7 = pd.read_excel('6yl4.xlsx')
xlist7 = xlist7.iloc[:,200:600]
xlist7 = xlist7.values  #converting to numpy for flexibility
ylist7 = ylist7.values  #converting to numpy for flexibility
te7 = pd.read_csv('TE443.csv')
strow7 = list(te7)
te7.loc[-1] = strow7  # adding a row
te7.index = te7.index + 1  # shifting index
te7 = te7.sort_index()  # sorting by index
te7 = te7.iloc[:,1]
te7 = te7.values
te7 = list(map(float, te7))

xtensor7 = torch.zeros(5000,400,dtype=torch.float)
ytensor7 = torch.zeros(5000,400,dtype=torch.float)
tetensor7 = torch.zeros(5000,1,dtype=torch.float)
xmax7 = torch.zeros(5000,1,dtype = torch.float)
ymax7 = torch.zeros(5000,1,dtype = torch.float)

temax7 = torch.zeros(1,dtype = torch.float)
temax7 = torch.max(tetensor7)
tepred7 = torch.zeros(5000,1,dtype=torch.float)
tepred7 = torch.div(tetensor7,temax7)
ztensor7 = torch.cat((xtensor7,ytensor7),0)
ztensor7.size()

for i in range(5000):
  xtensor7[i] = torch.tensor(xlist7[i])
  ytensor7[i] = torch.tensor(ylist7[i])
  tetensor7[i] = torch.tensor(te7[i])  
xmax7 = torch.max(xtensor7)
ymax7 = torch.max(ytensor7)
xtensor7 = torch.div(xtensor7,xmax7)
ytensor7 = torch.div(ytensor7,ymax7)
ztensor7 = torch.cat((xtensor7,ytensor7),1)
ztensor7.size()

lagtensor7 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor7[i] = torch.FloatTensor([4.0,4.0,3.0])
output7 = torch.cat((tetensor7,lagtensor7),1)

netoutput = torch.cat((netoutput,output7),0)
netinput = torch.cat((netinput,ztensor7),0)


# In[24]:


xlist7 = pd.read_excel('6xl4.xlsx')
ylist7 = pd.read_excel('6yl4.xlsx')
xlist7 = xlist7.iloc[:,200:600]
xlist7 = xlist7.values  #converting to numpy for flexibility
ylist7 = ylist7.values  #converting to numpy for flexibility
te7 = pd.read_csv('TE443.csv')
strow7 = list(te7)
te7.loc[-1] = strow7  # adding a row
te7.index = te7.index + 1  # shifting index
te7 = te7.sort_index()  # sorting by index
te7 = te7.iloc[:,1]
te7 = te7.values
te7 = list(map(float, te7))

xtensor7 = torch.zeros(5000,400,dtype=torch.float)
ytensor7 = torch.zeros(5000,400,dtype=torch.float)
tetensor7 = torch.zeros(5000,1,dtype=torch.float)
xmax7 = torch.zeros(5000,1,dtype = torch.float)
ymax7 = torch.zeros(5000,1,dtype = torch.float)

temax7 = torch.zeros(1,dtype = torch.float)
temax7 = torch.max(tetensor7)
tepred7 = torch.zeros(5000,1,dtype=torch.float)
tepred7 = torch.div(tetensor7,temax7)
ztensor7 = torch.cat((xtensor7,ytensor7),0)
ztensor7.size()

for i in range(5000):
  xtensor7[i] = torch.tensor(xlist7[i])
  ytensor7[i] = torch.tensor(ylist7[i])
  tetensor7[i] = torch.tensor(te7[i])  
xmax7 = torch.max(xtensor7)
ymax7 = torch.max(ytensor7)
xtensor7 = torch.div(xtensor7,xmax7)
ytensor7 = torch.div(ytensor7,ymax7)
ztensor7 = torch.cat((xtensor7,ytensor7),1)
ztensor7.size()

lagtensor7 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor7[i] = torch.FloatTensor([4.0,4.0,3.0])
output7 = torch.cat((tetensor7,lagtensor7),1)

netoutput = torch.cat((netoutput,output7),0)
netinput = torch.cat((netinput,ztensor7),0)


# In[25]:


xlist8 = pd.read_excel('7xl3.xlsx')
ylist8 = pd.read_excel('7yl4.xlsx')
xlist8 = xlist8.iloc[:,200:600]
xlist8 = xlist8.values  #converting to numpy for flexibility
ylist8 = ylist8.values  #converting to numpy for flexibility
te8 = pd.read_csv('TE343.csv')
strow8 = list(te8)
te8.loc[-1] = strow8  # adding a row
te8.index = te8.index + 1  # shifting index
te8 = te8.sort_index()  # sorting by index
te8 = te8.iloc[:,1]
te8 = te8.values
te8 = list(map(float, te8))

xtensor8 = torch.zeros(5000,400,dtype=torch.float)
ytensor8 = torch.zeros(5000,400,dtype=torch.float)
tetensor8 = torch.zeros(5000,1,dtype=torch.float)
xmax8 = torch.zeros(5000,1,dtype = torch.float)
ymax8 = torch.zeros(5000,1,dtype = torch.float)

temax8 = torch.zeros(1,dtype = torch.float)
temax8 = torch.max(tetensor8)
tepred8 = torch.zeros(5000,1,dtype=torch.float)
tepred8 = torch.div(tetensor8,temax8)
ztensor8 = torch.cat((xtensor8,ytensor8),0)
ztensor8.size()

for i in range(5000):
  xtensor8[i] = torch.tensor(xlist8[i])
  ytensor8[i] = torch.tensor(ylist8[i])
  tetensor8[i] = torch.tensor(te8[i])  
xmax8 = torch.max(xtensor8)
ymax8 = torch.max(ytensor8)
xtensor8 = torch.div(xtensor8,xmax8)
ytensor8 = torch.div(ytensor8,ymax8)
ztensor8 = torch.cat((xtensor8,ytensor8),1)
ztensor8.size()

lagtensor8 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor8[i] = torch.FloatTensor([3.0,4.0,3.0])
output8 = torch.cat((tetensor8,lagtensor8),1)

netoutput = torch.cat((netoutput,output8),0)
netinput = torch.cat((netinput,ztensor8),0)


# In[26]:


xlist9 = pd.read_excel('8xl3.xlsx')
ylist9 = pd.read_excel('8yl4.xlsx')
xlist9 = xlist9.iloc[:,200:600]
xlist9 = xlist9.values  #converting to numpy for flexibility
ylist9 = ylist9.values  #converting to numpy for flexibility
te9 = pd.read_csv('TE344.csv')
strow9 = list(te9)
te9.loc[-1] = strow9  # adding a row
te9.index = te9.index + 1  # shifting index
te9 = te9.sort_index()  # sorting by index
te9 = te9.iloc[:,1]
te9 = te9.values
te9 = list(map(float, te9))

xtensor9 = torch.zeros(5000,400,dtype=torch.float)
ytensor9 = torch.zeros(5000,400,dtype=torch.float)
tetensor9 = torch.zeros(5000,1,dtype=torch.float)
xmax9 = torch.zeros(5000,1,dtype = torch.float)
ymax9 = torch.zeros(5000,1,dtype = torch.float)

temax9 = torch.zeros(1,dtype = torch.float)
temax9 = torch.max(tetensor9)
tepred9 = torch.zeros(5000,1,dtype=torch.float)
tepred9 = torch.div(tetensor9,temax9)
ztensor9 = torch.cat((xtensor9,ytensor9),0)
ztensor9.size()

for i in range(5000):
  xtensor9[i] = torch.tensor(xlist9[i])
  ytensor9[i] = torch.tensor(ylist9[i])
  tetensor9[i] = torch.tensor(te9[i])  
xmax9 = torch.max(xtensor9)
ymax9 = torch.max(ytensor9)
xtensor9 = torch.div(xtensor9,xmax9)
ytensor9 = torch.div(ytensor9,ymax9)
ztensor9 = torch.cat((xtensor9,ytensor9),1)
ztensor9.size()

lagtensor9 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor9[i] = torch.FloatTensor([3.0,4.0,4.0])
output9 = torch.cat((tetensor9,lagtensor9),1)

netoutput = torch.cat((netoutput,output9),0)
netinput = torch.cat((netinput,ztensor9),0)


# In[27]:


xlist10 = pd.read_excel('9xl3.xlsx')
ylist10 = pd.read_excel('9yl2.xlsx')
xlist10 = xlist10.iloc[:,200:600]
xlist10 = xlist10.values  #converting to numpy for flexibility
ylist10 = ylist10.values  #converting to numpy for flexibility
te10 = pd.read_csv('TE322.csv')
strow10 = list(te10)
te10.loc[-1] = strow10  # adding a row
te10.index = te10.index + 1  # shifting index
te10 = te10.sort_index()  # sorting by index
te10 = te10.iloc[:,1]
te10 = te10.values
te10 = list(map(float, te10))

xtensor10 = torch.zeros(5000,400,dtype=torch.float)
ytensor10 = torch.zeros(5000,400,dtype=torch.float)
tetensor10 = torch.zeros(5000,1,dtype=torch.float)
xmax10 = torch.zeros(5000,1,dtype = torch.float)
ymax10 = torch.zeros(5000,1,dtype = torch.float)

temax10 = torch.zeros(1,dtype = torch.float)
temax10 = torch.max(tetensor10)
tepred10 = torch.zeros(5000,1,dtype=torch.float)
tepred10 = torch.div(tetensor10,temax10)
ztensor10 = torch.cat((xtensor10,ytensor10),0)
ztensor10.size()

for i in range(5000):
  xtensor10[i] = torch.tensor(xlist10[i])
  ytensor10[i] = torch.tensor(ylist10[i])
  tetensor10[i] = torch.tensor(te10[i])  
xmax10 = torch.max(xtensor10)
ymax10 = torch.max(ytensor10)
xtensor10 = torch.div(xtensor10,xmax10)
ytensor10 = torch.div(ytensor10,ymax10)
ztensor10 = torch.cat((xtensor10,ytensor10),1)
ztensor10.size()

lagtensor10 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor10[i] = torch.FloatTensor([3.0,2.0,2.0])
output10 = torch.cat((tetensor10,lagtensor10),1)

netoutput = torch.cat((netoutput,output10),0)
netinput = torch.cat((netinput,ztensor10),0)


# In[28]:


xlist11 = pd.read_excel('10xl4.xlsx')
ylist11 = pd.read_excel('10yl4.xlsx')
xlist11 = xlist11.iloc[:,200:600]
xlist11 = xlist11.values  #converting to numpy for flexibility
ylist11 = ylist11.values  #converting to numpy for flexibility
te11 = pd.read_csv('TE444.csv')
strow11 = list(te11)
te11.loc[-1] = strow11  # adding a row
te11.index = te11.index + 1  # shifting index
te11 = te11.sort_index()  # sorting by index
te11 = te11.iloc[:,1]
te11 = te11.values
te11 = list(map(float, te11))

xtensor11 = torch.zeros(5000,400,dtype=torch.float)
ytensor11 = torch.zeros(5000,400,dtype=torch.float)
tetensor11 = torch.zeros(5000,1,dtype=torch.float)
xmax11 = torch.zeros(5000,1,dtype = torch.float)
ymax11 = torch.zeros(5000,1,dtype = torch.float)

temax11 = torch.zeros(1,dtype = torch.float)
temax11 = torch.max(tetensor11)
tepred11 = torch.zeros(5000,1,dtype=torch.float)
tepred11 = torch.div(tetensor11,temax11)
ztensor11 = torch.cat((xtensor11,ytensor11),0)
ztensor11.size()

for i in range(5000):
  xtensor11[i] = torch.tensor(xlist11[i])
  ytensor11[i] = torch.tensor(ylist11[i])
  tetensor11[i] = torch.tensor(te11[i])  
xmax11 = torch.max(xtensor11)
ymax11 = torch.max(ytensor11)
xtensor11 = torch.div(xtensor11,xmax11)
ytensor11 = torch.div(ytensor11,ymax11)
ztensor11 = torch.cat((xtensor11,ytensor11),1)
ztensor11.size()

lagtensor11 = torch.zeros(5000,3,dtype=torch.float)
for i in range(5000):
  lagtensor11[i] = torch.FloatTensor([4.0,4.0,4.0])
output11 = torch.cat((tetensor11,lagtensor11),1)

netoutput = torch.cat((netoutput,output11),0)
netinput = torch.cat((netinput,ztensor11),0)


# In[29]:


maxin = torch.max(netinput)
maxout = torch.max(netoutput)
netinput = torch.div(netinput,maxin)
netoutput = torch.div(netoutput,maxout)


# In[30]:


netinput.shape


# In[31]:


class Neural_Network(nn.Module):
  def __init__(self,):
    super(Neural_Network,self).__init__()
    self.inputSize = 800
    self.outputSize = 4
    self.hiddenSize = 15
    self.hiddenSize2 = 15
   
    
    #weights
    self.W1 = torch.randn(self.inputSize, self.hiddenSize)
    self.W2 = torch.randn(self.hiddenSize, self.hiddenSize2)
    self.W3 = torch.randn(self.hiddenSize2,self.outputSize)
    self.B1 = torch.randn(1,self.hiddenSize)
    self.B2 = torch.randn(
        
        1,self.hiddenSize2)
    self.B3 = torch.randn(1,self.outputSize)
    
  def forward(self,sampleztensor):
    self.z = torch.add(torch.matmul(sampleztensor,self.W1),self.B1)
    self.z2 = self.sigmoid(self.z)
    self.z3 = torch.add(torch.matmul(self.z2,self.W2),self.B2)
    self.z4 = self.sigmoid(self.z3)
    self.z5 = torch.add(torch.matmul(self.z4,self.W3),self.B3)
    o =self.sigmoid(self.z5)
    return o
  
  def sigmoid(self,s):
    return 1/(1+torch.exp(-s))
  
  def sigmoidPrime(self,s):
    return s * (1 -s)
  
  def backward(self,sampleztensor,output,o):
    self.o_error = 2*(o - output)
    self.o_delta = self.o_error * self.sigmoidPrime(o)
    self.z4_error = torch.matmul(self.o_delta,torch.t(self.W3))
    self.z4_delta = self.z4_error*self.sigmoidPrime(self.z4)
    self.z2_error = torch.matmul(self.z4_delta, torch.t(self.W2))
    self.z_delta = self.z2_error*self.sigmoidPrime(self.z2)
    sampleztensor = sampleztensor.numpy()
    sampleztensor = sampleztensor.reshape(1,800)
    sampleztensor = torch.from_numpy(sampleztensor)
    self.W1 = 0.99*self.W1 - 0.01*torch.matmul(torch.t(sampleztensor),self.z_delta)
    self.W2 = 0.99*self.W2 - 0.01*torch.matmul(torch.t(self.z2),self.z4_delta)
    self.W3 = 0.99*self.W3 -  0.01*torch.matmul(torch.t(self.z4), self.o_delta)
    self.B1 = 0.99*self.B1 - 0.01*self.z_delta
    self.B2 = 0.99*self.B2 - 0.01*self.z4_delta
    self.B3 = 0.99*self.B3-  0.01*self.o_delta

  def train(self,sampleztensor,output):
    o = self.forward(sampleztensor)
    self.backward(sampleztensor,output,o)
    
  def saveWeights(self, model):
    torch.save(model,"NN")
  
  def predict(self):
    print("predicted data based on trained weights:\n")
    print("Input: \n" + str(sampleztensor))

    print("Output: \n"+ str(self.forward(sampleztensor)))


# In[ ]:



NN = Neural_Network()
for i in range(70000):
  NN.train(netinput[i,:],netoutput[i,:])
  print(str(torch.mean((output-NN(sampleztensor))**2).detach().item()))
NN.saveWeights(NN)
NN.predict()

