import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
# Metodo de arvores randomicas

# Abri o arquivo
with open('credito.pkl', 'rb') as f:
    x_trei, x_teste, y_trei, y_teste = pickle.load(f)

print(x_trei.shape, y_trei.shape)

"""
Algoritimo dos pesos
adam
sgd
lbfgs

Funções de ativação
relu
logistic(sigmoid)
tanh
identity
"""
rede_neural_credit = MLPClassifier(max_iter=1500, verbose=True, tol=0.000000100,
                                   solver = 'adam', activation = 'logistic',
                                   hidden_layer_sizes = (13,13))
rede_neural_credit.fit(x_trei, y_trei)

previsoes = rede_neural_credit.predict(x_teste)

accuracy_score(y_teste, previsoes)

rede_neural_credit.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    rede_neural_credit, x_teste, y_teste
)

print(classification_report(y_teste, previsoes))






















































































