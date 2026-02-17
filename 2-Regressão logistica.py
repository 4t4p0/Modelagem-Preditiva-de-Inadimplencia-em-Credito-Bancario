import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Abri o arquivo
with open('credito.pkl', 'rb') as f:
    x_trei, x_teste, y_trei, y_teste = pickle.load(f)
    
logistic_credit = LogisticRegression(random_state=1)
logistic_credit.fit(x_trei, y_trei)

previsoes = logistic_credit.predict(x_teste)

print(accuracy_score(y_teste, previsoes))

# Atualização em 11/09/2025
logistic_credit.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    logistic_credit, x_teste, y_teste
)

print(classification_report(y_teste, previsoes))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    