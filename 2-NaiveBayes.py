from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from sklearn.metrics import ConfusionMatrixDisplay

# Metodo de Naive Bayes

"""
1 - Carrega a base de dados


"""
# Abri o arquivo
with open('credito.pkl', 'rb') as f:
    x_trei, x_teste, y_trei, y_teste = pickle.load(f)

# Cria a Gaussiana e  carrega ela
naive_credit_data = GaussianNB()
naive_credit_data.fit(x_trei, y_trei)

# Faz as previsões
previsoes = naive_credit_data.predict(x_teste)

# Calcula a acuracia
accuracy_score(y_teste, previsoes)

# Printa a acuracia
print(classification_report(y_teste, previsoes))

# Cria o grafico da acuracia
naive_credit_data.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    naive_credit_data, x_teste, y_teste
)



















































































































