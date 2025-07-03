import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from matplotlib.pyplot import xlabel
from sklearn import preprocessing

features = pd.read_csv("./data/temps.csv")
print(features)
print(f"数据：{features.head()},shape:{features.shape}")

years = features['year']
months = features['month']
days = features['day']
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year,month,day in zip(years,months,days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# dates = pd.to_datetime(dates)
print(dates[0:5])

fig,((axe1,axe2),(axe3,axe4)) = plt.subplots(2,2,figsize=(10,10))
fig.autofmt_xdate(rotation=45)

plt.style.use('fivethirtyeight')
axe1.plot(dates, features['actual'])
axe1.set_title('Actual')
axe2.plot(dates, features['temp_1'])
axe2.set_title('Temp 1')
axe3.plot(dates, features['temp_2'])
axe3.set_title('Temp 2')
axe4.plot(dates, features['friend'])
axe4.set_title('Friend')
plt.tight_layout(pad=2)
# fig.show()

print(features)
features = pd.get_dummies(features)
print(features.columns)
print(type(features['actual']))

labels = np.array(features['actual'])
print(labels)

features = features.drop(['actual'], axis=1)
print(features.columns)

feature_list = list(features.columns)
features = np.array(features)
print(type(features))

