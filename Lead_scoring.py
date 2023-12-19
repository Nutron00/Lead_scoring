# Importing the basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # **2. Reading the dataset**
# 
# This is crucial for EDA as it provides the initial step to understand the data structure, format, and overall content, enabling subsequent analysis




df = pd.read_csv('Leads.csv')
df.head()





df.shape


# ## **2.1 Basic Statistics of the Numerical Variables**
# 
# This offers insights into the dispersion, and shape of the numerical features, aiding in identifying patterns, outliers, and potential areas for further investigation




df.describe()





df.info()


# ## **2.2 Checking for the Null Values within the dataset**
# 
# This is important to ensure `data completeness` and reliability, as missing values can impact analysis accuracy and inform decisions on data imputation or exclusion.




(df.isnull().mean() * 100).sort_values(ascending=False)





df.isnull().sum().plot.bar()


# ## **2.3 Analysing the values present in the dataset**
# 
# Analyzing dataset values is vital for EDA, providing insights into `variable distributions` and patterns, aiding in the identification of `outliers` and relationships.




for i in df.columns:
    name = df[i].unique()
    print(i, name)
    print("*" * 50)


# ##### ***Evaluating the count of values assigned to each data point in columns with null values.***
# 




null_var = ['Country', 'Specialization', 'How did you hear about X Education', 'What is your current occupation', 'What matters most to you in choosing a course', 'Tags', 'Lead Quality', 'Lead Profile', 'City']





for i in null_var:
    print(i)
    print('-'*10)
    print(df[i].value_counts())
    print('*'*90)
    print()





df['What is your current occupation'] = df['What is your current occupation'].replace('Select', np.nan)





df['What is your current occupation'] = df['What is your current occupation'].fillna('None')





df['What is your current occupation'].value_counts()


# #### ***Dropping columns with either a single value or an excessive number of null values, thus removing columns with more than 25% null values since 'select' is considered equivalent to null.***




drop_col = ['Prospect ID', 'Lead Number', 'Magazine', 'Lead Quality', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
           'Asymmetrique Activity Score', 'Asymmetrique Profile Score',
           'Receive More Updates About Our Courses',
           'Update me on Supply Chain Content',
           'Get updates on DM Content',
            'What matters most to you in choosing a course',
           'I agree to pay the amount through cheque',
           'Country',
           'How did you hear about X Education',
           'Specialization',
           'Tags',
           'Lead Profile',
           'City']






for col in drop_col:
    df = df.drop(col, axis = 1)





df.shape





for i in df.columns:

    print(df.groupby(i)['Converted'].sum())
    print('-'*15)
    print()





df.isnull().sum()





df.shape


# ##### ***Removing the rows that contains null values.***
# 




df = df.dropna(subset = df.columns.to_list())
df.shape





df.info()





df.head()





sns.pairplot(df)





for i in df.columns:
    j = 1
    plt.subplot(14,2,j)
    plt.figure(figsize = (6,6))
    print(df.groupby(i)['Converted'].sum().plot.barh())
    j += 1
    plt.show()





change_feat = ['Do Not Email', 'Do Not Call', 'Search', 'Newspaper Article', 'X Education Forums', 'Newspaper', 'Digital Advertisement',
              'Through Recommendations', 'A free copy of Mastering The Interview', ]


# ##### ***Transforming `'Yes/No'` to `'0/1'` for binary categorical variables enhances compatibility with numerical operations, facilitating streamlined analysis in subsequent code..***
# 




def convert(x):
    return x.map({'Yes':1, 'No':0})

df[change_feat] = df[change_feat].apply(convert)





df.head()


# ## **2.4 Creating Dummy Variables for Categorical Columns**
# 
# Creating dummy variables converts categorical data into a format `compatible with machine learning`, facilitating effective model training.




origin = pd.get_dummies(df['Lead Origin'], prefix = 'origin', drop_first = True)

df = pd.concat((df, origin), axis = 1)





df.drop('Lead Origin', axis =1, inplace = True)





df.head()





source = pd.get_dummies(df['Lead Source'], prefix = 'source', drop_first = True)

df = pd.concat((df, source), axis = 1)
df.drop('Lead Source', axis =1, inplace = True)





df.head()





activity = pd.get_dummies(df['Last Activity'], prefix = 'last_activity', drop_first = True)

df = pd.concat((df, activity), axis = 1)
df.drop('Last Activity', axis =1, inplace = True)





df.head()





notable_activity = pd.get_dummies(df['Last Notable Activity'], prefix = 'Notable_Activity', drop_first = True)

df = pd.concat((df, notable_activity), axis = 1)
df.drop('Last Notable Activity', axis =1, inplace = True)





special = pd.get_dummies(df['What is your current occupation'], prefix = 'occupation')
special = special.drop('occupation_None', axis = 1)
df = pd.concat((df, special), axis = 1)
df.drop('What is your current occupation', axis =1, inplace = True)
df.head()


# ## **2.5 Splitting Data into `Testing Data Set` and `Training Data Set`**
# 
# Splitting data into testing and training sets is essential for model evaluation and performance assessment in machine learning. 
# 
# **This process ensures that the model is trained on `one subset of the data` and tested on another, `providing a reliable measure of its generalization capabilities`.**




from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df,train_size = 0.7, random_state = 100)





df_train.shape





df_test.shape





X_train = df_train.drop('Converted', axis = 1)
y_train = df_train['Converted']





X_train.shape





y_train = y_train.values.reshape(-1)





y_train.shape


# ####  ***Scaling the dataset.***
# 
# Scaling the dataset involves `normalizing or standardizing the numeric features`, ensuring they are on a consistent scale. This is crucial for many machine learning algorithms that are sensitive to the magnitude of input variables, promoting improved model performance and convergence.
# 




from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()





X_train[X_train.columns.to_list()] = scaler.fit_transform(X_train[X_train.columns.to_list()])





X_train.describe()





# plt.figure(figsize = (15,15))
# sns.heatmap(X_train.corr(), annot = True, cmap = 'RdYlGn')


# ## **2.6 Running the `first` training model**
# 




import statsmodels
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train)
logml = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
logml.fit().summary()


# ## **2.7 `Recursive Feature Elimination`**
# 
# Recursive Feature Elimination (RFE) is a technique used for feature selection. It recursively removes features, fits the model, and ranks the features based on their importance.
# 




# selecting top features using RFE

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(estimator = logreg, n_features_to_select = 25)
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns, rfe.support_,rfe.ranking_))





col = X_train.columns[rfe.support_]





X_train_sm = sm.add_constant(X_train[col])
logreg = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
model = logreg.fit()
model.summary()


# ## **2.8 Variance Inflation Factor**
# 
# The `Variance Inflation Factor (VIF)` is used to identify the presence of ***`multicollinearity`*** in a regression analysis. It measures how much the variance of an estimated regression coefficient increases if your predictors are correlated.




from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['features'] = X_train_sm.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm.values,i) for i in range(X_train_sm.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif





# let's drop some columns based on its p-value high p-value means the variable is not significant.
col1 = col.drop(['Newspaper', 'occupation_Housewife'])





col1





X_train_sm = sm.add_constant(X_train[col1])
logreg = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
model = logreg.fit()
model.summary()





# checking variance inflation factor (vif)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['features'] = X_train_sm.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm.values,i) for i in range(X_train_sm.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif





col2 = col1.drop(['origin_Lead Add Form'])





X_train_sm = sm.add_constant(X_train[col2])
logreg = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
model = logreg.fit()
model.summary()





# checking variance inflation factor (vif)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['features'] = X_train_sm.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm.values,i) for i in range(X_train_sm.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif





X_train_sm = sm.add_constant(X_train[col2])
logreg = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
model = logreg.fit()
model.summary()





# checking features that contribute most towards the lead getting converted

data = X_train_sm.drop('const', axis = 1)
data['Converted'] = y_train
plt.figure(figsize = (20,15))
sns.heatmap(data.corr(), annot = True, cmap = 'RdYlGn')


# ## **3.Prediction**
# 
# The `Variance Inflation Factor (VIF)` is used to identify the presence of ***`multicollinearity`*** in a regression analysis. It measures how much the variance of an estimated regression coefficient increases if your predictors are correlated.




# prediction

y_train_pred = model.predict(X_train_sm).values.reshape(-1)
y_train_pred





# creating dataframe of actual and pred probability
y_train_pred_df = pd.DataFrame({'Actual':y_train, 'prob': y_train_pred})
y_train_pred_df.head()





# Improting evaluation metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix





# Defining the roc function
def draw_roc(actual, probs):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate = False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize = (5,5))
    plt.plot(fpr,tpr, label = 'Roc curve (area = % 0.2f)' % auc_score)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [ 1- True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc = 'lower right')
    plt.show()
    
    return None





# plotting roc plot
draw_roc(y_train_pred_df.Actual, y_train_pred_df.prob)





# creating column with optimal cutoff point
numbers = [float(x)/10 for x in range(10)] + [0.35]
for i in numbers:
    y_train_pred_df[i] = y_train_pred_df.prob.map(lambda x:1 if x> i else 0)

y_train_pred_df.head()





# calculating accuracy, sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame(columns = ['prob', 'accuracy', 'sensi', 'speci', 'preci'])

num = [ 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for i in num:
    cm = metrics.confusion_matrix(y_train_pred_df.Actual, y_train_pred_df[i])
    total = sum(sum(cm))
    accuracy = (cm[0,0] + cm[1,1])/ total
    speci = cm[0,0]/(cm[0,0] + cm[0,1])
    sensi = cm[1,1]/(cm[1,0] + cm[1,1])
    preci = cm[1,1]/(cm[0,1] + cm[1,1])
    cutoff_df.loc[i] = [i,accuracy, sensi, speci, preci]




print(cutoff_df)





# plotting the probabilities

cutoff_df.plot.line(x = 'prob', y = ['accuracy', 'sensi', 'speci', 'preci'])
plt.show()




# Here we will choose 40% probability  above this people will be predicted as potential lead.
# Making predictions on the test set

X_test = df_test.drop('Converted', axis = 1)
y_test = df_test['Converted']





print(X_test.shape,y_test.shape)




# scaling 
X_test[X_test.columns.to_list()] = scaler.transform(X_test[X_test.columns.to_list()])

X_test = X_test[col2]




X_test.head()





# predicting y
X_test_sm = sm.add_constant(X_test)
y_test_pred = model.predict(X_test_sm)
y_test_pred = y_test_pred.values.reshape(-1)
y_test_pred





# Creating Dataframe
y_test_df = pd.DataFrame({'Actual':y_test, 'Probability': y_test_pred})
y_test_df['ID'] = y_test_df.index
y_test_df.index = np.arange(y_test_df.shape[0])
y_test_df['predicted'] = y_test_df.Probability.map(lambda x:1 if x > .4 else 0)
y_test_df.head()





# Evaluation on the test dataset
confusion2 = metrics.confusion_matrix(y_test_df.Actual, y_test_df.predicted)
confusion2


# ## **4. Accuracy, Sensitivity, Specificity & Precision**
# 
# **Accuracy**: Overall correctness of the model, calculated as the ratio of correctly predicted instances to the total instances.
# 
# **Sensitivity (Recall)**: The proportion of actual positive instances that were correctly identified by the model. It's calculated as the ratio of true positives to the sum of true positives and false negatives.
# 
# **Specificity**: The proportion of actual negative instances that were correctly identified by the model. It's calculated as the ratio of true negatives to the sum of true negatives and false positives.
# 
# **Precision**: The accuracy of positive predictions made by the model. It's calculated as the ratio of true positives to the sum of true positives and false positives.




# Accuracy score
metrics.accuracy_score(y_test_df.Actual, y_test_df.predicted)





# sensitivity or recall
sen = confusion2[1,1] / (confusion2[1,1] + confusion2[1,0])
sen





# specificity
spec = confusion2[0,0] / (confusion2[0,0] + confusion2[0,1])
spec





# precision
preci = confusion2[1,1]/ (confusion2[0,1] + confusion2[1,1])
preci



