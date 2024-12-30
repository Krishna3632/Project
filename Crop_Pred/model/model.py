import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data=pd.read_csv("C://Users//ks520//Desktop//New folder//Dataset//crop.csv")

x=data.iloc[:,:-1] #for extracting all the columns except crop columns(Features)
y=data.iloc[:,-1]  #for extracting only the crop column(Label)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

model=RandomForestClassifier()

model.fit(x_train,y_train)

prediction=model.predict(x_test)

pickle.dump(model, open("model.pkl", "wb"))

