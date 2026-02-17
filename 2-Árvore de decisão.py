from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

# Metodo de arvores de decisão

# Abri o arquivo
with open('credito.pkl', 'rb') as f:
    x_trei, x_teste, y_trei, y_teste = pickle.load(f)
    
arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state = 0)
arvore_credit.fit(x_trei, y_trei)

previsoes = arvore_credit.predict(x_teste)

print(accuracy_score(y_teste, previsoes))

# Cria o grafico da acuracia
arvore_credit.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    arvore_credit, x_teste, y_teste
)























