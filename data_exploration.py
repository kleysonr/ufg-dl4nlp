import pandas as pd

# Le arquivo de treinamento
train = pd.read_csv('/data/datasets/mlc/train.csv')

# Visualiza primeiras linhas
train.head()

# Estatisticas basicas do dataset
#                                     title   label_quality      language     category
# count                          20.000.000      20.000.000    20.000.000   20.000.000
# unique                         19.988.405               2             2        1.588
# top       Microscopio Biologico Binocular      unreliable    portuguese        PANTS
# freq                                    2      18.815.755    10.000.000       35.973
train.describe()

# Conta numero de categorias
# Name: category, Length: 1588, dtype: int64
train['category'].value_counts()

# Conta numero de registro por label_quality
# unreliable    18.815.755
# reliable       1.184.245
train['label_quality'].value_counts()

# Conta numero de registros por language
# portuguese    10.000.000
# spanish       10.000.000
train['language'].value_counts()

# Verificar como esta a distribuicao de unreliable/reliable para cada language
# language   label_quality                   
# portuguese reliable        693318 ( 6.93%)
#            unreliable     9306682 (93.07%)
# spanish    reliable        490927 ( 4.91%)
#            unreliable     9509073 (95.09%)
train.groupby(['language','label_quality']).count()




