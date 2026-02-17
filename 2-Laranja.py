import Orange
from collections import Counter


# Abri o arquivo

base_credit = Orange.data.Table('Credito orange.csv')

majority = Orange.classification.MajorityLearner()
previsoes = Orange.evaluation.testing.TestOnTestData(base_credit, base_credit, [majority])

print(Orange.evaluation.CA(previsoes))

Counter(str(registro.get_class()) for registro in base_credit)

























