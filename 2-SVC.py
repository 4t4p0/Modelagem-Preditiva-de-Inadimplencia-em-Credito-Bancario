import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
# Metodo de arvores randomicas

# Abri o arquivo
with open('credito.pkl', 'rb') as f:
    x_trei, x_teste, y_trei, y_teste = pickle.load(f)
    
svm_credit = SVC(kernel='rbf', random_state=1, C = 2.0) # 2 -> 4
svm_credit.fit(x_trei, y_trei)

previsoes = svm_credit.predict(x_teste)
    
accuracy_score(y_teste, previsoes)
    
print(accuracy_score(y_teste, previsoes))

# Cria o grafico da acuracia
svm_credit.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    svm_credit, x_teste, y_teste
)


#=====================================================
svm_credit = SVC(kernel='linear', random_state=1, C = 2.0) # 2 -> 4
svm_credit.fit(x_trei, y_trei)

previsoes = svm_credit.predict(x_teste)
    
accuracy_score(y_teste, previsoes)
    
print(accuracy_score(y_teste, previsoes))

# Cria o grafico da acuracia
svm_credit.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    svm_credit, x_teste, y_teste
)


#=====================================================
svm_credit = SVC(kernel='poly', random_state=1, C = 2.0) # 2 -> 4
svm_credit.fit(x_trei, y_trei)

previsoes = svm_credit.predict(x_teste)
    
accuracy_score(y_teste, previsoes)
    
print(accuracy_score(y_teste, previsoes))

# Cria o grafico da acuracia
svm_credit.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    svm_credit, x_teste, y_teste
)



#=====================================================
svm_credit = SVC(kernel='sigmoid', random_state=1, C = 2.0) # 2 -> 4
svm_credit.fit(x_trei, y_trei)

previsoes = svm_credit.predict(x_teste)
    
accuracy_score(y_teste, previsoes)
    
print(accuracy_score(y_teste, previsoes))

# Cria o grafico da acuracia
svm_credit.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    svm_credit, x_teste, y_teste
)



#=====================================================
svm_credit = SVC(kernel='precomputed', random_state=1, C = 2.0) # 2 -> 4
svm_credit.fit(x_trei, y_trei)

previsoes = svm_credit.predict(x_teste)
    
accuracy_score(y_teste, previsoes)
    
print(accuracy_score(y_teste, previsoes))

# Cria o grafico da acuracia
svm_credit.fit(x_trei, y_trei)
ConfusionMatrixDisplay.from_estimator(
    svm_credit, x_teste, y_teste
)






