import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('D:\\Datascience\\Modules\\ML_DL_NLP\\4. Decision-Trees-and-Random-Forests\\Social_Network_Ads.csv')
print(dataset.info())

print(dataset['Purchased'].unique())
sns.heatmap(dataset.isnull())
plt.show()

dataset.drop(['Gender','User ID'], axis=1, inplace=True)

sns.heatmap(dataset.corr(),annot=True)
plt.show()

# Standardize the variables
scaler = StandardScaler()
scaler.fit(dataset.drop('Purchased',axis=1))
scaled_features = scaler.transform(dataset.drop('Purchased',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=['Age', 'EstimatedSalary'])
df_feat.head()

# Train and test the data
X_train, X_test, y_train, y_test = train_test_split(df_feat,dataset['Purchased'],
                                                    test_size=0.30,random_state=30)

# Randomforestclassifier
rfc = RandomForestClassifier()

# Initial variable declare for Randomforestclassifier
random_grid={'n_estimators':[x for x in np.arange(100,1100,100)],
    'max_features':['auto','sqrt'],
    'max_depth':[x for x in np.arange(5,30,6)],
    'min_samples_split':[x for x in np.arange(2,11)],
    'min_samples_leaf':[x for x in np.arange(1,11)],
    'criterion':['gini','entropy'],
    'bootstrap':[True,False]
}

print(random_grid)

# Optimize the values in RandomizedSearchCV
rf_random=RandomizedSearchCV(estimator=rfc,param_distributions=random_grid,n_iter=20,n_jobs=-1,cv=10,verbose=5)

rf_random.fit(X_train,y_train)

print(rf_random.best_params_)
print(rf_random.best_score_)

# Final hyperparameters for Randomforestclassifier
rfc_1 = RandomForestClassifier(n_estimators= 900,
 min_samples_split= 4,
 min_samples_leaf= 8,
 max_features= 'sqrt',
 max_depth= 23,
 criterion= 'gini',
 bootstrap= True)

rfc_1.fit(X_train,y_train)
predictions=rfc_1.predict(X_test)

# confusionmatrix
cm = confusion_matrix(y_test, predictions)
print(cm)
# Accuracy score
accuracy_score(y_test, predictions)