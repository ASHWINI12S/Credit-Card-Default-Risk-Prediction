import pandas as pd
df=pd.read_csv(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\Data\rawdata\rawdata.csv')

import os
os.makedirs(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\Data\cleanddata',exist_ok=True)

df.to_csv(r'C:\Users\ASHWINI\OneDrive - ATCS\Desktop\ML Project\Data\cleanddata\cleaned.csv',index=False)



