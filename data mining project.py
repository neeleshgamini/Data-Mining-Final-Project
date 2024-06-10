


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("bank.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.isna().sum()


# In[7]:


df.info()


# In[8]:


df.dtypes


# In[9]:


df.describe()


# In[10]:


df['y'].value_counts()


# In[11]:


for a in list(df.columns):
    n = df[a].unique()
    if len(n)<30:
        print(a)
        print(n)
    else:
        print(a + ': ' +str(len(n)) + ' unique values')


# Graphical Representation of Numerical Features

# In[12]:


cols_num = ['campaign', 'pdays','previous','age','balance']


# In[13]:


df[cols_num].head() 


# In[14]:


fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'age', data =  df[cols_num])
ax.set_xlabel('Age', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Age Count Distribution', fontsize=15)


# In[15]:


fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'age', data = df[cols_num], orient = 'v', ax = ax1)
ax1.set_xlabel('People Age', fontsize=15)
ax1.set_ylabel('Age', fontsize=15)
ax1.set_title('Age Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

sns.distplot(df[cols_num]['age'])
ax2.set_xlabel('Age', fontsize=15)
ax2.set_ylabel('Occurence', fontsize=15)
ax2.set_title('Age x Ocucurence', fontsize=15)
ax2.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout()


# In[16]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'previous', data = df[cols_num])
ax.set_xlabel('Previous', fontsize=16)
ax.set_ylabel('Number', fontsize=16)
ax.set_title('Previous', fontsize=16)
ax.tick_params(labelsize=16)


# In[17]:


df[cols_num].isnull().sum()


# Categorical Features

# In[18]:


cols_cat = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month', 'poutcome']


# In[19]:


df[cols_cat].isnull().sum()


# One-Hot Encoding

# In[20]:


cols_cat = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month', 'poutcome']
df[cols_cat]
cols_new_cat=pd.get_dummies(df[cols_cat],drop_first = False)
cols_new_cat.head(5)


# In[21]:


cols_new_cat.columns


# Graphical Representation of Categorical Features

# In[22]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'education', data = df[cols_cat])
ax.set_xlabel('Education Receieved', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.set_title('Education', fontsize=16)
ax.tick_params(labelsize=16)


# In[23]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'marital', data = df[cols_cat])
ax.set_xlabel('Marital Status', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.set_title('Marital', fontsize=16)
ax.tick_params(labelsize=16)


# In[24]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'job', data = df[cols_cat])
ax.set_xlabel('Types of Jobs', fontsize=16)
ax.set_ylabel('Number', fontsize=16)
ax.set_title('Job', fontsize=16)
ax.tick_params(labelsize=16)


# In[25]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'poutcome', data = df[cols_cat])
ax.set_xlabel('Marketing Campaign', fontsize=16)
ax.set_ylabel('Number of Previous Outcomes', fontsize=16)
ax.set_title('poutcome', fontsize=16)
ax.tick_params(labelsize=16)


# In[26]:


fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,8))
sns.countplot(x = 'default', data = df[cols_cat], ax = ax1, order = ['no', 'unknown', 'yes'])
ax1.set_title('Default', fontsize=15)
ax1.set_xlabel('')
ax1.set_ylabel('Count', fontsize=15)
ax1.tick_params(labelsize=15)

sns.countplot(x = 'housing', data = df[cols_cat], ax = ax2, order = ['no', 'unknown', 'yes'])
ax2.set_title('Housing', fontsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('Count', fontsize=15)
ax2.tick_params(labelsize=15)

sns.countplot(x = 'loan', data = df[cols_cat], ax = ax3, order = ['no', 'unknown', 'yes'])
ax3.set_title('Loan', fontsize=15)
ax3.set_xlabel('')
ax3.set_ylabel('Count', fontsize=15)
ax3.tick_params(labelsize=15)


# In[27]:


df = pd.concat([df,cols_new_cat], axis = 1)


# In[28]:


cols_all_cat=list(cols_new_cat.columns)


# In[29]:


df[cols_all_cat].head()


# In[30]:


df[cols_num+cols_all_cat].isna().sum()


# In[31]:


df['y'].value_counts()


# In[32]:


df['y'] = df['y'].map({'no':0,'yes':1})


# In[33]:


cols_input = cols_num + cols_all_cat
New_df = df[cols_input + ['y']]


# In[34]:


cols_input


# In[35]:


len(cols_input)


# In[36]:


New_df.head(5)


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


x = New_df.drop(['y'],axis=1)
y = New_df['y']


# In[39]:


from sklearn.preprocessing  import StandardScaler


# In[40]:


scaler=StandardScaler()


# In[41]:


scaled_data=scaler.fit_transform(x)


# In[42]:


x_train,x_test,y_train,y_test=train_test_split(scaled_data,y,test_size=0.2)


# In[43]:


print(y_train.shape)


# In[44]:


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
def print_report(y_actual, y_pred, thresh):
    
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    f1 = 2 * (precision * recall) / (precision + recall)
   
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print('f1:%.3f'%f1)
    print(' ')
    return auc, accuracy, recall, precision, f1


# In[45]:


thresh = 0.5


# KNN

# In[46]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors = 100)
knn = knn.fit(x_train, y_train)
pred=knn.predict(x_test)


# In[47]:


from sklearn.metrics import confusion_matrix
confusion_knn = confusion_matrix(y_test,pred)
sns.heatmap(confusion_knn,annot = True)


# In[48]:


y_train_preds = knn.predict_proba(x_train)[:,1]
y_valid_preds = knn.predict_proba(x_test)[:,1]

print('KNN')
print('Training:')
knn_train_auc, knn_train_accuracy, knn_train_recall, \
    knn_train_precision, knn_train_f1 = print_report(y_train,y_train_preds, thresh)
print('Testing:')
knn_valid_auc, knn_valid_accuracy, knn_valid_recall, \
    knn_valid_precision, knn_valid_f1 = print_report(y_test,y_valid_preds, thresh)


# Decision Tree

# In[49]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 10, random_state = 42)
DT = tree.fit(x_train, y_train)
pred_tree=DT.predict(x_test)


# In[50]:


from sklearn.metrics import confusion_matrix
confusion_tree = confusion_matrix(y_test,pred_tree)
sns.heatmap(confusion_tree,annot = True)


# In[51]:


y_train_preds = tree.predict_proba(x_train)[:,1]
y_valid_preds = tree.predict_proba(x_test)[:,1]

print('Decision Tree')
print('Training:')
tree_train_auc, tree_train_accuracy, tree_train_recall, tree_train_precision, tree_train_f1 =print_report(y_train,y_train_preds, thresh)
print('Testing:')
tree_valid_auc, tree_valid_accuracy, tree_valid_recall, tree_valid_precision, tree_valid_f1 = print_report(y_test,y_valid_preds, thresh)


# Random Forest

# In[52]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth = 6, random_state = 42)
RF = rf.fit(x_train, y_train)
pred_rf=RF.predict(x_test)


# In[53]:


from sklearn.metrics import confusion_matrix
confusion_rf = confusion_matrix(y_test,pred_rf)
sns.heatmap(confusion_rf,annot = True)


# In[54]:


y_train_preds = rf.predict_proba(x_train)[:,1]
y_valid_preds = rf.predict_proba(x_test)[:,1]

print('Random Forest')
print('Training:')
rf_train_auc, rf_train_accuracy, rf_train_recall, rf_train_precision, rf_train_f1 = print_report(y_train,y_train_preds, thresh)
print('Testing:')
rf_valid_auc, rf_valid_accuracy, rf_valid_recall, rf_valid_precision, rf_valid_f1 = print_report(y_test,y_valid_preds, thresh)


# Gradient Boosting

# In[55]:


from sklearn.ensemble import GradientBoostingClassifier
gbc =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=3, random_state=42)
GBC = gbc.fit(x_train, y_train)
pred_gbc=GBC.predict(x_test)


# In[56]:


from sklearn.metrics import confusion_matrix
confusion_gbc = confusion_matrix(y_test,pred_gbc)
sns.heatmap(confusion_gbc,annot = True)


# In[57]:


y_train_preds = gbc.predict_proba(x_train)[:,1]
y_valid_preds = gbc.predict_proba(x_test)[:,1]

print('Gradient Boosting Classifier')
print('Training:')
gbc_train_auc, gbc_train_accuracy, gbc_train_recall, gbc_train_precision, gbc_train_f1 = print_report(y_train,y_train_preds, thresh)
print('Testing:')
gbc_valid_auc, gbc_valid_accuracy, gbc_valid_recall, gbc_valid_precision, gbc_valid_f1 = print_report(y_test,y_valid_preds, thresh)


# Through this project, we created a machine learning model that is able to predict how likely clients will subscribe to a bank term deposit. The best model was Random Forest classifier. The model's performance is 90.1%.

# In[ ]:




