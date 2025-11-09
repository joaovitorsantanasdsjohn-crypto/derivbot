import os
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier


MODEL_FILE = "model.pkl"


class MLModel:
def __init__(self):
# tries to load pre-trained model; otherwise init incremental model
if os.path.exists(MODEL_FILE):
with open(MODEL_FILE, "rb") as f:
self.model = pickle.load(f)
print("Modelo carregado de model.pkl")
self.classes_ = getattr(self.model, "classes_", ["CALL","PUT"])
else:
print("Nenhum model.pkl encontrado — inicializando SGDClassifier para treinamento incremental")
self.model = SGDClassifier(max_iter=1000, tol=1e-3)
# inicial fit dummy to enable partial_fit
X0 = np.array([[0,0,0,0],[1,1,1,1]])
y0 = np.array(["CALL","PUT"])
self.model.partial_fit(X0, y0, classes=["CALL","PUT"])


def predict_proba_and_label(self, features):
# SGDClassifier não tem predict_proba por padrão — usamos decision_function -> convert to probability via sigmoid
X = np.array(features).reshape(1,-1)
try:
