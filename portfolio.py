import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ads = pd.read_csv('advertising.csv')
print(ads.info())
print(ads.describe())

# -------- EXPLORATORY DATA -----------
#sns.set_style('darkgrid')

f= plt.figure(figsize=(4,4))
sns.distplot(ads['Age'],bins=30,kde=False)
plt.title('Histogram of Age')
plt.savefig('AgeHist.jpg',dpi=300,bbox_inches='tight')
plt.tight_layout()
plt.close()

f = plt.figure(figsize=(4,4))
sns.jointplot(ads['Age'],ads['Area Income'],data=ads)
plt.savefig('Joint_AgeAreaIncome.jpg',dpi=300,bbox_inches='tight')
plt.tight_layout()
plt.close()

f = plt.figure(figsize=(10,10))
sns.heatmap(ads.corr(),cmap='coolwarm',annot=True)
plt.savefig('heatmap.jpg',dpi=300,bbox_inches='tight')
plt.tight_layout()
plt.close()

f = plt.figure(figsize=(10,10))
sns.pairplot(ads, vars=['Age','Daily Time Spent on Site','Area Income'],hue='Clicked on Ad')
plt.savefig('pairplot.jpg',dpi=300,bbox_inches='tight')
plt.tight_layout()
plt.close()

# -------- Logistic Regression ------
# define X and y databases to drive logistic model
X = ads.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'],axis=1)
y = ads['Clicked on Ad']

# design test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
logm = LogisticRegression().fit(X_train,y_train)

# ----- Predictions and Evaluation -----
predictions = logm.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
