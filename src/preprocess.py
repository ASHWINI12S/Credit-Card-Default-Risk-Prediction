import pandas as pd
df = pd.read_csv(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\Data\cleanddata\cleaned.csv')
df.head()
df['default.payment.next.month'].value_counts()
df[df.duplicated()]
df.info()
from imblearn.over_sampling import SMOTE

df.drop('ID',axis=1,inplace=True)

x = df.drop('default.payment.next.month',axis=1)
y = df['default.payment.next.month']

x,y = SMOTE().fit_resample(X=x,y=y)
y.value_counts()

import os
os.makedirs(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\Data\preprocessdata',exist_ok=True)

x.to_csv(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\Data\preprocessdata\x.csv',index=False)

y.to_csv(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\Data\preprocessdata\y.csv',index=False)
