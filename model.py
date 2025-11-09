import os
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier

MODEL_FILE = "model.pkl"


class MLModel:
    def __init__(self):
        # tenta carregar modelo pré-treinado; caso contrário, inicia modelo incremental
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                self.model = pickle.load(f)
            print("Modelo carregado de model.pkl")
            self.classes_ = getattr(self.model, "classes_", ["CALL", "PUT"])
        else:
            print("Nenhum model.pkl encontrado — inicializando SGDClassifier para treinamento incremental")
            self.model = SGDClassifier(max_iter=1000, tol=1e-3)
            # inicialização mínima para habilitar partial_fit
            X0 = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
            y0 = np.array(["CALL", "PUT"])
            self.model.partial_fit(X0, y0, classes=["CALL", "PUT"])

    def predict_proba_and_label(self, features):
        # SGDClassifier não possui predict_proba por padrão
        # convertemos decision_function em probabilidade via sigmoid
        X = np.array(features).reshape(1, -1)
        try:
            score = self.model.decision_function(X)
            prob = 1 / (1 + np.exp(-score))  # sigmoid
            if prob.size == 1:
                proba_call = prob[0]
                proba_put = 1 - proba_call
                label = "CALL" if proba_call > 0.5 else "PUT"
                return float(max(proba_call, proba_put)), label
            else:
                idx = np.argmax(prob)
                label = self.model.classes_[idx]
                return float(prob[idx]), label
        except Exception as e:
            print("Erro em predict_proba_and_label:", e)
            return 0.5, "CALL"

    def partial_train(self, X, y):
        try:
            self.model.partial_fit(X, y)
            with open(MODEL_FILE, "wb") as f:
                pickle.dump(self.model, f)
        except Exception as e:
            print("Erro ao treinar modelo:", e)
