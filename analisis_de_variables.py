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
	print(index)
	print(row['Test']) 
	if (row['Test'] > 0.2) or (row ['Train'] > 0.2):
		eliminar.append(index)
#En la variable eliminar estan los nombres de las variables que deben ser directamente eliminadas
#Dentro de las variables a eliminar encontramos que la variable de Alley NA no indica desconocido, es un posible valor más de los posibles a tomar
#Esa variable debe seguir estando en nuestro conjunto
print(eliminar)
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

#El resto de variables pueden rellenarse con la media
for var in numericas:
	if features[var].isnull().sum() > 0:
		features[var] = features[var].fillna(features[var].mean())
#El resto ce variables nomnales se rellenan con el valor más frecuente
for var in nominales:
	if features[var].isnull().sum() > 0:
		features[var] = features[var].fillna(features[var].mode()[0])
"""Una vez que la tabla de datos está en el formato correcto vamos a estudiar la correlación 
de las variables con el precio. Las variables que presenten una correlación baja se descartarán 
ya que lo único que van a hacer es hacer que nuestro modelo se impreciso. 
Si se imputan demasiadas variables perderemos información valiosa y el modelo volverá a ser impreciso.
Sacando un Heatmap se puede ver la correlación de las variables"""

#train_labels = np.log(train_labels)#La transformación logarítmica de los datos los aproxima a una distribución normal
complete = features.loc['train']#Solo se usan las entradas de entrenamiento
complete = pd.concat([complete,train_labels],axis=1)#Se adjunta la columna de precios de nuevo

correlationPlot = complete.corr()#Mantiene la matriz de correlación en un DataFrame
f,ax = plt.subplots(figsize=(12,9))#Configuración del tamaño de la imagen
sns.heatmap(correlationPlot,vmax=.8,square=True)#Crea el heatmap con los valores de correlación
plt.yticks(rotation=0)#cambia el eje de las etiquetas del gráfico para que se vean bien
plt.xticks(rotation=90)#cambia el eje de las etiquetas del gráfico para que se vean bien
plt.show()#Muestra el gráfico 
f.savefig('Heatmap.png')#Guarda el gráfico en un archivo

"""La matriz de correlación muestra la correlación entre dos variables de forma que los valores 
más claros muestran que dos variables tienen una correlación alta
El siguiente paso del análisis es buscar que variables muestran una correlación alta entre sí y eliminar 
una de esas variables, ya que es información redundante y puede eliminarse. Otra manera de enfocar el problema
es que usar dos variables correlacionadas puede ayudar a sofocar el efecto del ruido en una variable.
En primer lugar es necesario descubrir que variables son las que determinan el precio de la vivienda usando la correlación.
"""
#Crear la lista de variables con correlación alta con el precio de la vivienda
"""Inciso:
	calcular la correlación antes de aplicar la escala logaritmica a los datos
	tiene sentido, pues el coeficiente de correlación de Pearson no varía con 
	la escala y el origen. Además solo nos sirve para hacer una aproximación
	hacia que variables usar o no en el algoritmo. Después si será necesario 
	hacer que las variables tengan una distribución normalizada. 
"""
HighCorrelation = []
for index, row in correlationPlot.iterrows(): 
	if (row['SalePrice'] >= 0.5) or (row ['SalePrice'] <= -0.5):
		HighCorrelation.append(index)
		print(row['SalePrice'])
print("total de variables: "+str(len(HighCorrelation)))
print(HighCorrelation)
"""Ahora hay que examniar las variables nominales que se tendrán en cuenta
Para hacer este análisis se va a usar una gráfica que exprese la relación entre
el precio y el valor de la vivienda."""
complete = features.loc['train']
complete = pd.concat([complete,train_labels],axis=1)
malas = [#'MSSubClass',
	'LandContour',
	'LandSlope',
	#'RoofStyle',
	#'RoofMatl',
	'Exterior2nd',
	#'Exterior1st',
	'MasVnrType',
	'BsmtExposure',
	'Functional',
	'YrSold']
##################################
#malas = ['Utilities', 'RoofMatl','Heating','Functional']
for var in malas:
	data = pd.concat([complete[var],complete['SalePrice']],axis=1)
	f,ax = plt.subplots(figsize=(12,9))
	fig = sns.boxplot(x=var,y="SalePrice",data=data)
	fig.axis(ymin=0,ymax=800000)
	plt.xticks(rotation=90)
	f.savefig(str(var)+'_Price.png')
"""
aparentemente malas variables:
	LandContour
	LandScope
	RoofStyle
	RoofMatl
	Exterior2nd
	Exterior1st
	MasVnrType
	BsmtExposure
	Functional
	YrSold
"""
"""Analisis con PCA"""
