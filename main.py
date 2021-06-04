import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

print('============================Таблица============================')
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(data)
print('-------------------Задание 1-------------------')
guy_live = 0
guy_dead = 0
women_live = 0
women_dead = 0
for index, row in data.iterrows():  # Чтобы перебрать строки, мы применяем функцию iterrows
    row[4] = str(row[4])
    row[1] = str(row[1])
    if row[1] == "1" and row[4] == "male":
        guy_live = guy_live + 1
    elif row[1] == "0" and row[4] == "male":
        guy_dead = guy_dead + 1
    elif row[1] == "1" and row[4] == "female":
        women_live = women_live + 1
    else:
        women_dead = women_dead + 1
print('Соотношение выживших мужчин к погибшим: ', guy_live / guy_dead)
print('Соотношение выживших женщин к погибшим: ', women_live / women_dead)
print('-------------------Задание 1-------------------')
print('Другим способом о котором я узнал слишком поздно')
#  groupby используется для разделения и выделения некоторой части данных из всего набора данных
#  value_counts возвращает серию, содержащую количество уникальных значений
#  normalize - если вы хотите проверить частоту вместо подсчетов.
print(data.groupby(["Sex"])["Survived"].value_counts(normalize=True))

print('-------------------Задание 2-------------------')
describe_fields = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
print("Статистика по числовым полям для мужчин: ")
male = data[data["Sex"] == "male"][describe_fields].describe()
print(male)
print('===============================================')
print("Статистика по числовым полям для женщин: ")
female = data[data["Sex"] == "female"][describe_fields].describe()
print(female)

print('-------------------Задание 3-------------------')
c = 0
q = 0
s = 0
c1 = 0
q1 = 0
s1 = 0
for index, row in data.iterrows():  # Чтобы перебрать строки, мы применяем функцию iterrows
    row[11] = str(row[11])
    row[1] = str(row[1])
    if row[1] == "1" and row[11] == 'C':
        c = c + 1
    elif row[1] == "1" and row[11] == 'Q':
        q = q + 1
    elif row[1] == "1" and row[11] == 'S':
        s = s + 1
    elif row[1] == "0" and row[11] == 'C':
        c1 = c1 + 1
    elif row[1] == "0" and row[11] == 'Q':
        q1 = q1 + 1
    elif row[1] == "0" and row[11] == 'S':
        s1 = s1 + 1
print('Умерло из C:', c, 'Умерло из Q:', q, 'Умерло из S:', s)
print('Возможно и есть какая-то зависимость?')
print('Выжили из C:', c1, 'Выжили из Q:', q1, 'Выжили из S:', s1)
print('Хотя это может быть обусловлено тем, что из разных портов выезжало разное количество')

print('-------------------Задание 3-------------------')
print('Аналогично 1 номеру...')
print(data.groupby(["Embarked"])["Survived"].value_counts(normalize=True))

print('-------------------Задание 4-------------------')
print(data["Name"].value_counts()[:10])

print('-------------------Задание 5-------------------')
print(data.isnull().sum())
dict_value = {}
print('============================Таблица============================')
header = ["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
for key in header:
    dict_value[key] = data[key].median()
data.fillna(dict_value)
print(data)

print('-------------------Задание 6-------------------')
print('=======================================Таблица=======================================')
X = data.drop(['Survived', 'Sex',
                'Embarked', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1).values
Y = data['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
pipe.fit(X_train, y_train)
print("Точность определения:   ", pipe.score(X_test, y_test)*100, "% результаты для пассажиров будут такими")
column = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
_dict = {}
for key in column:
    _dict[key] = test[key].median()
test.fillna(_dict, inplace=True)
X_test = pd.concat([test[column]], axis=1)
pred = pipe.predict(X_test)
print(pred)