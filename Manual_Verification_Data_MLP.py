"**********************Maunal verification of the modal*******************"
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
seed=6
np.random.seed(seed)

Data=pd.read_csv("Data.csv")
#print(Data.head(50))
#print("Abhi")
#print(Data.head(1))
#print(Data.head())
"*********************Features Engineering***************************************"
color={'Cool':1,'Neutral':2,'Warm':3}
music={'Rock':1,'Hip':2,'Jazz/Blues':3,'Pop':4,'Electronic':5,'Hip hop':6,'R&B and soul':7,'Folk/Traditional':8}
beverage={"Beer":1,"Doesn't drink":2,"Other":3,"Vodka":4,"Whiskey":5,"Wine":6}
drink={"Coca Cola/Pepsi":1,"Fanta":2,"7UP/Sprite":3,"Other":4}
gender={"F":1,"M":0}
favtcolor=Data['Favorite Color'].map(color)
#print(favtcolor)
favtmusic=Data['Favorite Music Genre'].map(music)
#print(favtmusic)
favtbeverage=Data['Favorite Beverage'].map(beverage)
#print(favtbeverage)
favtdrink=Data['Favorite Soft Drink'].map(drink)
#print(favtdrink)
Y=Data['Gender'].map(gender)
#print(Gender)
X=pd.concat([favtcolor,favtmusic,favtbeverage,favtdrink], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33,random_state=seed)
#print(newdata.head(5))
"*****************************************Building model*******************************"
model = Sequential()
model.add(Dense(16, input_dim=4, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y,validation_data=(X_test,y_test),epochs=150, batch_size=10,  verbose=2)
"****************************Evaluating Modal*****************************"
#evaluate the model
scores=model.evaluate(X_test,y_test)
#result=model.predict(X_test)


#print("\n%s:%2f%%"%(model.metrics_names[1], "Accuracy",scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
