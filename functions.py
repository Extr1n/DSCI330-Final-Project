import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing 
from sklearn.ensemble import RandomForestRegressor

dfall = pd.read_csv("data/MentalHealthSurvey.csv")


df = dfall.drop(['university', 'degree_level', 'degree_major', 'stress_relief_activities'], axis=1)

df =df.replace(to_replace="Male", value=1)
df =df.replace(to_replace="Female", value=-1)
df =df.replace(to_replace="1st year", value=1)
df =df.replace(to_replace="2nd year", value=2)
df =df.replace(to_replace="3rd year", value=3)
df =df.replace(to_replace="4th year", value=4)
df =df.replace(to_replace="0.0-0.0", value=0)
df =df.replace(to_replace="1.5-2.0", value=1)
df =df.replace(to_replace="2.0-2.5", value=2)
df =df.replace(to_replace="2.5-3.0", value=3)
df =df.replace(to_replace="3.0-3.5", value=4)
df =df.replace(to_replace="3.5-4.0", value=5)
df =df.replace(to_replace="Off-Campus", value=-1)
df =df.replace(to_replace="On-Campus", value=1)
df =df.replace(to_replace="Yes", value=1)
df =df.replace(to_replace="No", value=-1)
df =df.replace(to_replace="No Sports", value=0)
df =df.replace(to_replace="1-3 times", value=1)
df =df.replace(to_replace="4-6 times", value=2)
df =df.replace(to_replace="7+ times", value=3)
df =df.replace(to_replace="4-6 hrs", value=5)
df =df.replace(to_replace="2-4 hrs", value=3)
df =df.replace(to_replace="7-8 hrs", value=8)


x = df.drop(['depression', 'anxiety', 'isolation', 'future_insecurity', 'gender', 'age', 'academic_year'], axis=1)

depression = df['depression']
anxiety = df['anxiety']
isolation = df['isolation']
insecurity = df['future_insecurity']

dx_train, dx_test, dy_train, dy_test = train_test_split(x, depression, test_size=0.15, random_state=40)
ax_train, ax_test, ay_train, ay_test = train_test_split(x, anxiety, test_size=0.15, random_state=40)
ix_train, ix_test, iy_train, iy_test = train_test_split(x, isolation, test_size=0.15, random_state=40)
fx_train, fx_test, fy_train, fy_test = train_test_split(x, insecurity, test_size=0.15, random_state=40)


model_depression = LinearRegression()
rf_depression = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

model_depression.fit(dx_train, dy_train)
rf_depression.fit(dx_train, dy_train)

model_anxiety = LinearRegression()
rf_anxiety = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

model_anxiety.fit(ax_train, ay_train)
rf_anxiety.fit(ax_train, ay_train)

model_isolation = LinearRegression()
rf_isolation = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

model_isolation.fit(ix_train, iy_train)
rf_isolation.fit(ix_train, iy_train)

model_insecurity = LinearRegression()
rf_insecurity = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

model_insecurity.fit(fx_train, fy_train)
rf_insecurity.fit(fx_train, fy_train)

models = [model_depression, model_anxiety, model_isolation, model_insecurity]
y = [depression, anxiety, isolation, insecurity]
y_names = ["depression", "anxiety", "isolation", "future_insecurity"]

x_names = x.columns


# Model Coeffecients

def get_model_coeffecients():
    for i in range(len(x.columns)):
        print()
        print(x.columns[i])
        for model in models:
            print(round(model.coef_[i], 7), end="\t")

# Exploratory Data Analysis Box Plots


x = ["gender", "age", "average_sleep", "cgpa", "residential_status", "campus_discrimination", "academic_pressure", "social_relationships", "financial_concerns", "sports_engagement"]

def get_box_plot(i):
    sns.set_theme(style="darkgrid")
    for m in y:
        plt.figure(figsize=(3,3))
        sns.boxplot(x=dfall[i], y=m)
        plt.show()

# Exploratory Data Analysis Heatmaps

def get_heatmap(m):
  pivot = (dfall.groupby(m)[y_names].mean())

  plt.figure(figsize=(12, 5))

  sns.heatmap(pivot, annot=True, cmap="flare", fmt=".2f", cbar_kws={'label': 'Mean Value'})
  plt.title('Mental Health Factors by ' + m)
  plt.ylabel(m)
  plt.show()
