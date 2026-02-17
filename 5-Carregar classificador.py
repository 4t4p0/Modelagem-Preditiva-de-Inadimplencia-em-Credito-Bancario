import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Abri o arquivo
with open('credito.pkl', 'rb') as f:
    x_trei, x_teste, y_trei, y_teste = pickle.load(f)

X_credit = np.concatenate((x_trei, x_teste), axis = 0)
y_credit = np.concatenate((y_trei, y_teste), axis = 0)

# Árvore de decisão
with open(r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\arvore_finalizado.sav', 'rb') as f:
    classificador_arvore = pickle.load(f)

# Random Forest
with open(r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\random_forest_finalizado.sav', 'rb') as f:
    classificador_random_forest = pickle.load(f)

# KNN
with open(r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\knn_finalizado.sav', 'rb') as f:
    classificador_knn = pickle.load(f)

# Regressão Logística
with open(r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\logistica_finalizado.sav', 'rb') as f:
    classificador_logistica = pickle.load(f)

# SVM
with open(r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\svm_finalizado.sav', 'rb') as f:
    classificador_svm = pickle.load(f)

# Rede Neural
with open(r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\rede_neural_finalizado.sav', 'rb') as f:
    classificador_rede_neural = pickle.load(f)
    






