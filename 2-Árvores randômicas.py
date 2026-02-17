import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Metodo de arvores randomicas

# Abri o arquivo
with open('credito.pkl', 'rb') as f:
    x_trei, x_teste, y_trei, y_teste = pickle.load(f)
    
random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state = 0)
random_forest_credit.fit(x_trei, y_trei)

previsoes = random_forest_credit.predict(x_teste)

print(accuracy_score(y_teste, previsoes))

# Cria o grafico da acuracia
random_forest_credit.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    random_forest_credit, x_teste, y_teste
)
