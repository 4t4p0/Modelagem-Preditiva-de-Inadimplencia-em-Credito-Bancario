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

# Treinamento dos classificadores

classificador_arvore = DecisionTreeClassifier(
    criterion='entropy',
    min_samples_leaf=10,
    min_samples_split=10,
    splitter='random'
)
classificador_arvore.fit(X_credit, y_credit)

classificador_random_forest = RandomForestClassifier(
    criterion='gini',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100
)
classificador_random_forest.fit(X_credit, y_credit)

classificador_knn = KNeighborsClassifier(p=1, n_neighbors=10)
classificador_knn.fit(X_credit, y_credit)

classificador_logistica = LogisticRegression(
    C=1.0,
    solver='lbfgs',
    tol=0.0001
)
classificador_logistica.fit(X_credit, y_credit)

classificador_svm = SVC(kernel='rbf', C=2.0, probability=True)
classificador_svm.fit(X_credit, y_credit)

classificador_rede_neural = MLPClassifier(
    activation='relu',
    batch_size=100,
    solver='adam'
)
classificador_rede_neural.fit(X_credit, y_credit)

pickle.dump(
    classificador_rede_neural,
    open(
        r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\rede_neural_finalizado.sav',
        'wb'
    )
)

pickle.dump(
    classificador_arvore,
    open(
        r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\arvore_finalizado.sav',
        'wb'
    )
)

pickle.dump(
    classificador_svm,
    open(
        r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\svm_finalizado.sav',
        'wb'
    )
)

pickle.dump(
    classificador_random_forest,
    open(
        r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\random_forest_finalizado.sav',
        'wb'
    )
)

pickle.dump(
    classificador_knn,
    open(
        r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\knn_finalizado.sav',
        'wb'
    )
)

pickle.dump(
    classificador_logistica,
    open(
        r'C:\Users\lucas\OneDrive\Área de Trabalho\Machine learning\5-Classificadores salvos\logistica_finalizado.sav',
        'wb'
    )
)






