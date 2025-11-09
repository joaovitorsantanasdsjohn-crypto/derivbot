import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Assuma um CSV com colunas: rsi, close, upper, lower, target (CALL/PUT)
data = pd.read_csv("historico_candles.csv")
X = data[["rsi","close","upper","lower"]]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=50, max_depth=6)
clf.fit(X_train, y_train)
print("Acuracia:", clf.score(X_test,y_test))
with open("model.pkl","wb") as f:
    pickle.dump(clf,f)
