# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:40:53 2017

@author: Sergio
"""

#Analisis de variables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
import warnings
##########################################
# Prints R2 and RMSE scores
def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))
############################################
# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)
###########################################
	
#Ignorar los warnings
warnings.filterwarnings('ignore')
#Lectura de los datos
#En train se guandan los datos con los que se entrenará al modelo
train = pd.read_csv('train.csv')
#En test se guarda el conjunto de datos para el test
test = pd.read_csv('test.csv')

#Primero hay que eliminar las varibles que tengan un número alto de valores perdidos
#El número de valores perdidos de cada conjunto en cada variable
NAs = pd.concat([train.isnull().sum()/1460, test.isnull().sum()/1459], axis=1, keys=['Train', 'Test'])
#print(NAs)
#Eliminar todas las variables que tengan más de un 0.2 de valores perdidos
eliminar = []
nvars = 0
for index, row in NAs.iterrows():
	#print(index)
	#print(row['Test']) 
	if (row['Test'] > 0.2) or (row ['Train'] > 0.2):
		eliminar.append(index)
#En la variable eliminar estan los nombres de las variables que deben ser directamente eliminadas
#Dentro de las variables a eliminar encontramos que la variable de Alley NA no indica desconocido, es un posible valor más de los posibles a tomar
#Esa variable debe seguir estando en nuestro conjunto
eliminar.remove('Alley')
eliminar.remove('FireplaceQu')#Sucede lo mismo que con Alley

train.drop(eliminar,axis=1, inplace=True)
test.drop(eliminar,axis=1, inplace=True)

"""
Ahora es necesario un análisis más profundo de las variables.
En primer lugar encontramos algunas variables que parecen tener una representación
numérica, como por ejemplo 'MSSubClass' o 'OverallCond'. 
Al leer la documentación sobre que información aportan las variables
encontramos que OverallCond aunque sea una variable aparentemente nominal 
expresa cosas que son medibles como la calidad, es decir muestra una puntuación entre 1 y 10
"""
#Variables numéricas que deben ser transformadas a string
test['MSSubClass'] = test['MSSubClass'].astype(str)
train['MSSubClass'] = train['MSSubClass'].astype(str)

test['YrSold'] = test['YrSold'].astype(str)
train['YrSold'] = train['YrSold'].astype(str)

test['MoSold'] = test['MoSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)

#Variables categóricas que deben ser numéricas, ya que expresan puntuación
#El lógico pensar que aumentar la puntuación en algo hace efecto directo en el precio final
ExterQualvalues = {'ExterQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
ExterCondvalues = {'ExterCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}

BsmQualvalues = {'BsmtQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}

BsmCondvalues = {'BsmtCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,}}

HeatingQCvalues = {'HeatingQC':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}

KitchenQualvalues = {'KitchenQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}

FireplaceQuvalues = {'FireplaceQu':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}

GarageCondvalues = {'GarageCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
GarageQualvalues = {'GarageQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}

PoolQCvalues = {'PoolQC':{'Ex':4,'Gd':3,'TA':2,'Fa':1}}

#Reemplazar los valores en las tablas
train.replace(ExterQualvalues,inplace=True)
train.replace(ExterCondvalues,inplace=True)
train.replace(BsmQualvalues,inplace=True)
train.replace(BsmCondvalues,inplace=True)
train.replace(HeatingQCvalues,inplace=True)
train.replace(KitchenQualvalues,inplace=True)
train.replace(FireplaceQuvalues,inplace=True)
train.replace(GarageCondvalues,inplace=True)
train.replace(GarageQualvalues,inplace=True)
train.replace(PoolQCvalues,inplace=True)

test.replace(ExterQualvalues,inplace=True)
test.replace(ExterCondvalues,inplace=True)
test.replace(BsmQualvalues,inplace=True)
test.replace(BsmCondvalues,inplace=True)
test.replace(HeatingQCvalues,inplace=True)
test.replace(KitchenQualvalues,inplace=True)
test.replace(FireplaceQuvalues,inplace=True)
test.replace(GarageCondvalues,inplace=True)
test.replace(GarageQualvalues,inplace=True)
test.replace(PoolQCvalues,inplace=True)

#Ahora tenemos todas las variables con un tipo de dato 'correcto'
#Cuantas variables de cada tipo tenemos
train_labels = train.pop('SalePrice')
train_labels = np.log(train_labels)
features = pd.concat([train, test], keys=['train', 'test'])

enteras = features.dtypes[features.dtypes == 'int64'].index
flotantes = features.dtypes[features.dtypes == 'float64'].index
nominales = features.dtypes[features.dtypes == 'object'].index
#Se pasa a formato lista para su uso
ent = []
for var in enteras:
	ent.append(var)
flot = []
for var in flotantes:
	flot.append(var)
nom = []
for var in nominales:
	nom.append(var)

numericas = ent+flot

#Ahora es necesario rellenar los valores perdidos de cada variable.
"""En algunas de las variables que han sido transformadas a numéricas
NAN no expresa que el dato no exista, sino que expresa puntuación 0"""

features['BsmtQual'] = features['BsmtQual'].fillna(0)
features['BsmtCond'] = features['BsmtCond'].fillna(0)
features['FireplaceQu'] = features['FireplaceQu'].fillna(0)
features['GarageQual'] = features['GarageQual'].fillna(0)
features['GarageCond'] = features['GarageCond'].fillna(0)
#features['PoolQC'] = features['PoolQC'].fillna(0)
#El resto de variables pueden rellenarse con la media
for var in numericas:
	if features[var].isnull().sum() > 0:
		features[var] = features[var].fillna(features[var].mean())
#Algunas variables nominales no usan NA como valor perdido, sino como otro valor distinto
features['Alley'] = features['Alley'].fillna('NotAlley')
features['BsmtFinType1'] = features['BsmtFinType1'].fillna('NoBasement')
features['BsmtFinType2'] = features['BsmtFinType2'].fillna('NoBasement')
features['GarageType'] = features['GarageType'].fillna('NoGarage')
features['GarageFinish'] = features['GarageFinish'].fillna('NoGarage')
#features['Fence'] = features['Fence'].fillna('NoFence')
#features['MiscFeature'] = features['MiscFeature'].fillna('None')

#El resto ce variables nomnales se rellenan con el valor más frecuente
for var in nominales:
	if features[var].isnull().sum() > 0:
		features[var] = features[var].fillna(features[var].mode()[0])
		
"""Una vez que la tabla de datos está en el formato correcto vamos a eliminar los datos
que no cumplan las condiciones de correlación
"""
complete = features.loc['train']#Solo se usan las entradas de entrenamiento
complete = pd.concat([complete,train_labels],axis=1)#Se adjunta la columna de precios de nuevo
correlationPlot = complete.corr()#Mantiene la matriz de correlación en un DataFrame
LowCorrelation = []#Almacena las variables de baja correlación
for index, row in correlationPlot.iterrows(): 
	if (row['SalePrice'] <= 0.0) and (row ['SalePrice'] >= 0.0):
		LowCorrelation.append(index)
		print(row['SalePrice'])
print("total de variables: "+str(len(LowCorrelation)))
print(LowCorrelation)
#LowCorrelation.remove('Id')
#Se eliminan las variables de baja correlación
features.drop(LowCorrelation,axis=1, inplace=True)
#Variables categóricas a eliminar
BadCategoric = ['Utilities', 'RoofMatl','Heating','Functional']
features.drop(BadCategoric,axis=1, inplace=True)
#estandarización de variables
#Volvemos a actualizar la lista de variables numéricas y nominales
enteras = features.dtypes[features.dtypes == 'int64'].index
flotantes = features.dtypes[features.dtypes == 'float64'].index
nominales = features.dtypes[features.dtypes == 'object'].index
#Se pasa a formato lista para su uso
ent = []
for var in enteras:
	ent.append(var)
flot = []
for var in flotantes:
	flot.append(var)
nom = []
for var in nominales:
	nom.append(var)

numericas = ent+flot
#Hay que soltar el ID
numericas.remove('Id')
#Tabla con solo las variables numéricas
numeric_features = features[numericas]
numeric_features = np.log(numeric_features)
#Tabla con las variables numéricas estandarizadas
numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()
#Dummies de las nominales
#Para las variables condición 1 y condición 2, al ser variables nominales y poder fusionarse se crea una nueva columna para cada 
#posible valor de la variable y añade el valor 1 si la instancia tenía ese valor en variable 1 ó 2 y 0 en caso contrario

#Lista de condiciones
conditions = set([x for x in features['Condition1']] + [x for x in features['Condition2']])
print (conditions)
#nuevas columnas de la tabla inicializadas a 0
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(conditions))),
                       index=features.index, columns=conditions)
#Pone a 1 las celdas correspondientes
for i, cond in enumerate(zip(features['Condition1'], features['Condition2'])):
    dummies.ix[i, cond] = 1
#Concatena con la tabla de datos
features = pd.concat([features, dummies.add_prefix('Condition_')], axis=1)
#Elimina las columnas de condition1 y condition2
features.drop(['Condition1', 'Condition2'], axis=1, inplace=True)

#Se procede de igual forma con Exterior1st y Exterioor2nd
exteriors = set([x for x in features['Exterior1st']] + [x for x in features['Exterior2nd']])
print(exteriors)
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(exteriors))),
                       index=features.index, columns=exteriors)
for i, ext in enumerate(zip(features['Exterior1st'], features['Exterior2nd'])):
    dummies.ix[i, ext] = 1
features = pd.concat([features, dummies.add_prefix('Exterior_')], axis=1)
features.drop(['Exterior1st', 'Exterior2nd',], axis=1, inplace=True)

#Como el resto de variables nominales no se pueden fusionar se hace lo mismo pero de forma automática
for col in features.dtypes[features.dtypes == 'object'].index:
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)

###############################################################################################################################
#Hasta aquí la tabla obtenida tiene todas las variables numéricas y una columna por cada valor posible en cada variable	nominal
###############################################################################################################################
### Copying features
features_standardized = features.copy()

### Replacing numeric features by standardized values
features_standardized.update(numeric_features_standardized)

### Splitting features
train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Splitting standardized features
train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Shuffling train sets
train_features_st, train_features, train_labels = shuffle(train_features_st, train_features, train_labels, random_state = 5)

### Splitting
x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)

'''
Elastic Net
'''
ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train_st, y_train_st)
train_test(ENSTest, x_train_st, x_test_st, y_train_st, y_test_st)    

# Average R2 score and standard deviation of 5-fold cross-validation
scores = cross_val_score(ENSTest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
Gradient Boosting
'''
GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)
train_test(GBest, x_train, x_test, y_train, y_test)

# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(GBest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Retraining models
GB_model = GBest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

## Getting our SalePrice estimation
Final_labels = (np.exp(GB_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2

## Saving to CSV
pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('submission.csv', index =False)