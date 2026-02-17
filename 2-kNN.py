import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Abri o arquivo
with open('credito.pkl', 'rb') as f:
    x_trei, x_teste, y_trei, y_teste = pickle.load(f)

knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
knn_credit.fit(x_trei, y_trei)

previsoes = knn_credit.predict(x_teste)

print(accuracy_score(y_teste, previsoes))

# Atualização em 11/09/2025
knn_credit.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    knn_credit, x_teste, y_teste
)

print(classification_report(y_teste, previsoes))
