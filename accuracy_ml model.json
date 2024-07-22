import pandas as pd
import numpy as np
data = pd.read_excel('/content/Load Demand Data3.xlsx')

!pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

pd.to_numeric(data.DEMAND, errors="ignore")

data.head()

data.isnull().sum()

data.dropna(inplace=True)

data.shape

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = data[['week 2', 'week 3', 'week 4', 'MA_X-4', 'dayOfWeek', 'weekend', 'holiday','Holiday_ID', 'hourOfDay', 'T2M_toc']]
y = data['DEMAND']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

pd.DataFrame(data)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared (R2) Score: {r2:.2f}')
