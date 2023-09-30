import pandas as pd
import random
import math
import numpy as np

#   Считывание, обработка и нормализация исходных данных
def normalize_data(path):
    data = pd.read_csv(path)
    data = data.drop(columns=['Отметка времени'])
    data['Занимаетесь спортом?'] = data['Занимаетесь спортом?'].fillna('Да')
    data['Административный округ'] = data['Административный округ'].fillna('Подмосковье')
    for column in data.loc[:,data.columns != 'Кофе или чай?']:
        data[column] = data[column].astype('category')
        data[column] = data[column].cat.codes
        data[column] = ((data[column] - data[column].min())
        /(data[column].max() - data[column].min()))
    return data

#   Получение ближайших k соседей к объекту
def get_neighbors(all,point, k = 3):
    distances = [(euclidean_distance(all[i],point), all[i])
                for i in range(len(all))]
    distances.sort()
    return distances[:k]

#   Функция евклидова расстояния между объектами
def euclidean_distance(p1,p2):
    distance = 0
    for i in range(1,len(p1)):
        distance += (p1[i] - p2[i])**2
    return distance**0.5

#   Оценка точности
def get_accuracy(test_data,test_predict):
    count = 0
    for i in range(len(test_data)):
        if test_data[i][0] == test_predict[i]:
            count += 1
    return count / len(test_data)

#   Классификация на основании данных о соседях
def classify(neighbors):
    answer = 0
    for i in neighbors:
        answer += (i[1][0] == 'Чай') - (i[1][0] == 'Кофе')
    if answer > 0:
        return 'Чай'
    else:
        return 'Кофе'

#   Прогноз класса на основании данных об объекте ДОПИСАТЬ
def prediction(all,point,nbrs_count):
    neighbors = get_neighbors(all,point,nbrs_count)
    answer = classify(neighbors)
    print(F'For a {point} class is {answer}')

#   Функция разбиения выборки на обучающее и тестовое множество
def data_split(data,test_perсent = 0.3):
    test = []
    test_len = math.ceil(len(data)*test_perсent)
    for i in range(test_len):
        index = random.randint(0,len(data)-1)
        test.append(data.pop(index))
    return (data,test)

#   Функция для анализа наилучшего количества соседей
def get_best_neighbors_count(accuracy_values):
    best_count = [0.0,0]
    for i in accuracy_values:
        if i[0] > best_count[0]:
            best_count = i
    return best_count

data = normalize_data("Data1.csv")
train, test = data_split(data.values.tolist(), 0.3)

accuracy_values = []

# Нечетное кол-во для избежания проблемы определения класса при равном кол-ве
for nbrs_count in range(3,len(train)-1,2):
    predicts = []
    for i in range(len(test)):
        neighbors = get_neighbors(train,test[i],nbrs_count)
        predicts.append(classify(neighbors))
    accuracy_values.append([get_accuracy(test,predicts),nbrs_count])
    print('{:.6f}'.format(accuracy_values[-1][0]),F'K = {nbrs_count}')
best_count = get_best_neighbors_count(accuracy_values)
print(F'Best neighbors count is {best_count[1]}. Accuracy is {best_count[0]}')