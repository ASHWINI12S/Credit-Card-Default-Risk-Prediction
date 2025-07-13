import pandas as pd
x = pd.read_csv(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\Data\preprocessdata\x.csv')
y = pd.read_csv(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\Data\preprocessdata\y.csv')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=.3,stratify=y)

x_train.shape, y_train.shape

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_scaled,y_train)
predict = lr.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))

from sklearn.tree import DecisionTreeClassifier
Dt = DecisionTreeClassifier()
Dt.fit(x_train_scaled,y_train)
predict = Dt.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))

from sklearn.ensemble import RandomForestClassifier
Rf = RandomForestClassifier()
Rf.fit(x_train_scaled,y_train)
predict = Rf.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))

from sklearn.ensemble import AdaBoostClassifier
Ad = AdaBoostClassifier()
Ad.fit(x_train_scaled,y_train)
predict = Ad.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))


from sklearn.ensemble import GradientBoostingClassifier
Gdb = GradientBoostingClassifier()
Gdb.fit(x_train_scaled,y_train)
predict = Gdb.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))

from sklearn.svm import SVC
svc = GradientBoostingClassifier()
svc.fit(x_train_scaled,y_train)
predict = svc.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))

import pickle

with open(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\models\models.pkl','wb') as file:
    pickle.dump(Rf,file)

with open(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\models\models.pkl','rb') as file:
    loaded_model =pickle.load(file)


loaded_model.predict(x_test_scaled)
