import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

#============MACHINE LEARNING============

"""
pd.set_option('display.max_columns', None) - Printa colunas cheias
np.set_printoptions(threshold=np.inf) - Printa colunas cheias

1 - Tratar valores inconsistentes
    a - valores nulos
    b - valores estranhos
    
    
    #++++++troca colunas++++++
    colunas = dados.columns.tolist()

    i, j = colunas.index('coluna_A'), colunas.index('coluna_B')
    colunas[i], colunas[j] = colunas[j], colunas[i]

    dados = dados[colunas]
    #++++++troca colunas++++++
    
    base.describe() - descreve a base
    base.isnull().sum() - vê a quantidade de valores nulos 
    base['Categoria'].fillna(base['Categoria'].mean(), inplace = True) - substitui os valores pela média
    pd.set_option('display.max_columns', None) - Exibe todas as colunas
   
2 - Criar as variaveis de analise e de previsão
    X = base_credit.iloc[:, 1:@@@ Última linha @@@].values - cria o array x
    y = base_credit.iloc[:, @@@ Última linha @@@].values - cria o array y 

    ==================== Para variaveis nominais ============================
    label_encoder_workclass = LabelEncoder()
    label_encoder_education = LabelEncoder()
    label_encoder_marital = LabelEncoder()
    X[:,1] = label_encoder_workclass.fit_transform(X[:,1])
    X[:,3] = label_encoder_education.fit_transform(X[:,3])
    X[:,5] = label_encoder_marital.fit_transform(X[:,5])
    
    onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5])], remainder='passthrough')
    x = onehotencoder_census.fit_transform(x).toarray()
    
    scaler_credit = StandardScaler() - escala os valores para terem o mesmo padrão
    X = scaler_credit.fit_transform(X)
    
3 - Cria um grupo de treinamento e um grupo de teste
    x_trei, x_teste, y_trei, y_teste = train_test_split(x, y, test_size = 0.25, random_state = 0)
    
"""

# Configuração do print
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)

# Lê a base de dados
dados = pd.read_csv('credit_risk_dataset.csv')

# Preenche os dados faltantes
dados['person_emp_length'].fillna(dados['person_emp_length'].mean(), inplace = True)
dados['loan_int_rate'].fillna(dados['loan_int_rate'].mean(), inplace = True)

# Troca as colunas
colunas = dados.columns.tolist()

i, j = colunas.index('loan_status'), colunas.index('cb_person_cred_hist_length')
colunas[i], colunas[j] = colunas[j], colunas[i]

dados = dados[colunas]


# Transforma panda em array
x = dados.iloc[:, 1:11].values
y = dados.iloc[:, 11].values

# Transforma as colunas nominais em numeros
label_encoder_person_home_ownership = LabelEncoder()
label_encoder_loan_intent = LabelEncoder()
label_encoder_loan_grade = LabelEncoder()
label_encoder_cb_person_default_on_file = LabelEncoder()
x[:,1] = label_encoder_person_home_ownership.fit_transform(x[:,1])
x[:,3] = label_encoder_loan_intent.fit_transform(x[:,3])
x[:,4] = label_encoder_loan_grade.fit_transform(x[:,4])
x[:,9] = label_encoder_cb_person_default_on_file.fit_transform(x[:,9])

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,4,9])], remainder='passthrough')
x = onehotencoder_census.fit_transform(x)

# Escalona as colunas
scaler_credit = StandardScaler()
x= scaler_credit.fit_transform(x)


# Cria a base de teste e de treinamento
x_trei, x_teste, y_trei, y_teste = train_test_split(x, y, test_size = 0.25, random_state = 0)

print(y_trei[1])


# Salva tudo em um arquivo pickle
with open('credito.pkl', mode = 'wb') as f:
    pickle.dump([x_trei, x_teste, y_trei, y_teste], f)





















































    