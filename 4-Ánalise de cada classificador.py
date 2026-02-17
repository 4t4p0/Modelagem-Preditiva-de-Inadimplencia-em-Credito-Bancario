import pandas as pd
from scipy.stats import shapiro
import seaborn as sns
import matplotlib.pyplot as plt

resultados = pd.read_excel(r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\4-Resultado vários testes.xlsx')

print(resultados.describe())

print(resultados.var())

(resultados.std() / resultados.mean()) * 100

# Criar cada conjunto de resultados (Series)
resultados_arvore = resultados['Arvore']
resultados_random_forest = resultados['Random forest']
resultados_knn = resultados['KNN']
resultados_logistica = resultados['Logistica']
resultados_svm = resultados['SVM']
resultados_rede_neural = resultados['Rede neural']

alpha = 0.05

# Teste de Shapiro-Wilk
print('Arvore:', shapiro(resultados_arvore))
print('Random Forest:', shapiro(resultados_random_forest))
print('KNN:', shapiro(resultados_knn))
print('Logistica:', shapiro(resultados_logistica))
print('SVM:', shapiro(resultados_svm))
print('Rede Neural:', shapiro(resultados_rede_neural))

# KDE plots
sns.displot(resultados_arvore, kind='kde')
plt.title('Árvore de Decisão')

sns.displot(resultados_random_forest, kind='kde')
plt.title('Random Forest')

sns.displot(resultados_knn, kind='kde')
plt.title('KNN')

sns.displot(resultados_logistica, kind='kde')
plt.title('Regressão Logística')

sns.displot(resultados_svm, kind='kde')
plt.title('SVM')

sns.displot(resultados_rede_neural, kind='kde')
plt.title('Rede Neural')

plt.show()

print(resultados.mean())



