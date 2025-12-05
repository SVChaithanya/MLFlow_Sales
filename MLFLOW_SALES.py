import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment('sales_data')
mlflow.sklearn.autolog()

df=pd.read_csv(r"C:\Users\surya\ML_Project\p1.csv")
df.drop(columns=['Order Date','Order ID','Ship Date'],inplace=True)
print(df.isna().sum())
print(df['Total Profit'].skew())
Q1=df['Total Profit'].quantile(0.25)
Q3=df['Total Profit'].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df=df[(df['Total Profit']>=lower) & (df['Total Profit']<=upper)]
print(df.head())
print(df.describe())
print(df.info())

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isna())
plt.show()

sns.histplot(df['Total Profit'])
plt.show()

sns.scatterplot(x='Country',y='Total Profit',hue='Order Priority',data=df)
plt.show()

from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,root_mean_squared_error


with mlflow.start_run(run_name='pipe_5_autolog'):
    X = df.drop(columns=['Total Profit'])
    y = df['Total Profit']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    num = ['Units Sold','Unit Price','Unit Cost','Total Revenue','Total Cost']
    cat = ['Region','Country','Item Type','Sales Channel','Order Priority']
    num_line = Pipeline([
    ('knn_imputer',KNNImputer(n_neighbors=5)),
    ('scaler',StandardScaler())
    ])
    cat_line = Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(handle_unknown='ignore'))
    ])
    pipe_1 = ColumnTransformer(transformers=[('num',num_line,num),('cat',cat_line,cat)])


    model= {
    'rf':RandomForestRegressor(),
    'lr':LinearRegression(),
    'knn':KNeighborsRegressor()
    }
    result = []

    for name,m in model.items():
        pipe_2=Pipeline([
            ('pipe',pipe_1),
            ('model',m)
        ])
        pipe_2.fit(X_train,y_train)
        s_pred=pipe_2.predict(X_test)
        r2 = r2_score(y_test,s_pred)
        rmse = root_mean_squared_error(y_test,s_pred)
        result.append([name,r2,rmse])
    print(pd.DataFrame(result,columns=['model','R2','RMSE']))


 
    pipe_3 = Pipeline([
        ('pipe',pipe_1),
        ('rf',RandomForestRegressor())
    ])
    param = {
        'rf__n_estimators':[50,100,150],
        'rf__max_depth':[2,3,4]
    }   


    RF = RandomizedSearchCV(pipe_3,param,scoring='r2',random_state=42,cv=3,n_iter=10)
    RF.fit(X_train,y_train)
    best_RF_model = RF.best_estimator_.named_steps['rf']
    print('RF:',RF.best_params_)


    pipe_4 = Pipeline([
        ('pipe',pipe_1),
        ('knn',KNeighborsRegressor())
    ])
    param_1 = {
        'knn__n_neighbors':[3,5,10]
    }


    KNN= RandomizedSearchCV(pipe_4,param_1,scoring='r2',random_state=42,cv=3,n_iter=10)
    KNN.fit(X_train,y_train)
    best_KNN_model = KNN.best_estimator_.named_steps['knn']
    print("KNN:",KNN.best_params_)


    stack = StackingRegressor(estimators=[('rf',best_RF_model),
                                      ('lr',LinearRegression()),
                                      ('knn',best_KNN_model)],
                          final_estimator=RandomForestRegressor())


    pipe_5 = Pipeline([
        ('pipe',pipe_1),
        ('stack',stack)
    ])
    pipe_5.fit(X_train,y_train)
    y_pred=pipe_5.predict(X_test)
    print('r2:',r2_score(y_test,y_pred))
    print('Rmse:',root_mean_squared_error(y_test,pipe_5.predict(X_train)))
    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(pipe_5, artifact_path="model", signature=signature)








