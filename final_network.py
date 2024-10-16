import random
import sys
import pickle

# Сторонние библиотеки
import numpy as np
import pandas as pd
import csv

dt = np.dtype(np.float64)
loss_arr = []
np.seterr(divide='ignore')


class Network(object):

    def __init__(self, sizes):
        """ Массив sizes содержит количество нейронов в соответствующих слоях. Так что, если мы хотим создать объект Network с двумя нейронами в первом слое, тремя нейронами во втором слое, и одним нейроном в третьем, то мы запишем это, как [2, 3, 1]. Смещения и веса сети инициализируются случайным образом с использованием распределения Гаусса с математическим ожиданием 0 и среднеквадратичным отклонением 1. Предполагается, что первый слой нейронов будет входным, и поэтому у его нейронов нет смещений, поскольку они используются только при подсчёте выходных данных. """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.t = [0] * sizes[-1]
        self.res_entropy = 100

        ######## new
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y, )
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # print('INIT PARAMS')
        print(f'self.num_layers = {self.num_layers},\n'
              f'self.sizes = {self.sizes}\n'
              f'self.biases = {self.biases[:1]}\n'
              f'self.weights = {self.weights[:1]}\n'
              f'type self.biases = {type(self.biases[0])}\n'
              f'type self.weights = {type(self.weights[0])}\n'

              f'---------------------')

    def feedforward(self, a):
        """Возвращает выходные данные сети, когда ``a`` - входные данные."""
        for b, w in zip(self.biases, self.weights):
            a = ReLu(a @ w + b)
            # a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, eta, test_data=None):
        """Обучаем сеть при помощи мини-пакетов и стохастического градиентного спуска. training_data – список кортежей "(x, y)", обозначающих обучающие входные данные и желаемые выходные. Остальные обязательные параметры говорят сами за себя. Если test_data задан, тогда сеть будет оцениваться относительно проверочных данных после каждой эпохи, и будет выводиться текущий прогресс. Это полезно для отслеживания прогресса, однако существенно замедляет работу. """

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            for i in range(n):
                x, y = training_data[i][0], training_data[i][1]
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)

                counter = 0
                for w, dw in zip(self.weights, delta_nabla_w):
                    self.weights[counter] = w - eta * dw
                    counter += 1

                counter = 0
                for b, db in zip(self.biases, delta_nabla_b):
                    self.biases[counter] = b - eta * db
                    counter += 1

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


    def backprop(self, x, y):
        """Вернуть кортеж ``(nabla_b, nabla_w)``, представляющий градиент для функции стоимости C_x.  ``nabla_b`` и ``nabla_w`` - послойные списки массивов numpy, похожие на ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape, dtype=dt) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=dt) for w in self.weights]

        # прямой проход
        activation = x
        activations = [x]  # список для послойного хранения активаций
        counter = 0
        zs = []  # список для послойного хранения z-векторов
        activation = np.matrix(activation)
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = activation @ w + b
            # print(f'z = {z}\n')
            zs.append(z)
            relu = ReLu(z)
            activation = relu
            activations.append(activation)
            # print(f'activations = {activations}\n')
            counter += 1

        # # Последний слой
        b = self.biases[-1]
        w = self.weights[-1]
        z = activation @ w + b
        zs.append(z)
        activation = self.softmax(z)
        activations.append(activation)
        counter += 1

        # обратный проход
        self.res_entropy = self.binary_cross_entropy(y, activations[-1])
        dE_dtlast = self.cost_derivative(activations[-1], y)
        dE_dt1 = dE_dtlast
        for i in range(2, len(activations)):
            dE_dW2 = activations[-i].T @ dE_dtlast
            dE_db2 = dE_dt1
            dE_dh1 = dE_dt1 @ self.weights[-i + 1].T
            dE_dt1 = np.array(dE_dh1, dtype=dt) * np.array(ReLu_deriv(zs[-i]),
                                                           dtype=dt)
            nabla_b[-i + 1] = dE_db2
            nabla_w[-i + i] = dE_dW2

        dE_dW1 = np.matrix(activations[0], dtype=dt).T @ dE_dt1
        dE_db1 = dE_dt1
        nabla_b[0] = dE_db1
        nabla_w[0] = dE_dW1

        loss_arr.append(self.res_entropy)
        return (nabla_b, nabla_w)

    def my_evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforward(x)), y) for (x, y) in
                       test_data]
        print(test_result)

    def evaluate(self, test_data):
        """Вернуть количество проверочных входных данных, для которых нейросеть выдаёт правильный результат. Выходные данные сети – это номер нейрона в последнем слое с наивысшим уровнем активации."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Вернуть вектор частных производных (чп C_x / чп a) для выходных активаций."""
        return (output_activations - y)

    def binary_cross_entropy(self, t, p):

        t = np.float_(t)
        p = np.float_(p).T
        return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))


    def min_max_scaler(self, data, new_min=-20, new_max=20):
        # Преобразуем данные в двумерный массив, если они одномерные
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Находим минимум и максимум в каждом столбце
        min_vals = np.min(data, axis=1, keepdims=True)
        max_vals = np.max(data, axis=1, keepdims=True)

        # Масштабируем данные
        scaled_data = new_min + (data - min_vals) / (max_vals - min_vals) * (
                new_max - new_min)
        return scaled_data

    def softmax(self, x):
        scaled_data = self.min_max_scaler(x)
        out = np.exp(scaled_data)
        return out / np.sum(out)


    def predict(self, x):
        t1 = x @ self.weights[0] + self.biases[0]
        h1 = ReLu(t1)
        t2 = h1 @ self.weights[1] + self.biases[1]
        z = self.softmax(t2)
        return z

    def calc_accuracy(self, data):
        correct = 0
        for i in range(len(data) - 1):
            x, y = data[i][0], data[i][1]
            z = self.predict(x)
            y_pred = np.argmax(z)
            res = np.argmax(y)
            if y_pred == res:
                correct += 1

        acc = correct / len(data)
        return acc





#### Разные функции
def sigmoid(z):
    """Сигмоида."""
    return 1.0 / (1.0 + np.exp(-z))


def ReLu(z):
    return np.maximum(z, 0)


def ReLu_deriv(z):
    return (z >= 0).astype(float)


def sigmoid_prime(z):
    """Производная сигмоиды."""
    sig = sigmoid(z)
    print(f'sig = {sig}\n')
    print(f'type(sig) = {type(sig)}\n')
    print(f'(1-sig) = {(1 - sig)}')
    return sig @ (1 - sig).transpose()


df = pd.read_csv(
    'D:/MLUniversity/work1/Dataset/BrowserLogos_64/final_output_64.csv')
# df = pd.read_csv('D:/MLUniversity/work1/Dataset/BrowserLogos/output.csv')

# sys.exit(0)

training_data = []
training_data2 = []
class_label = []
pixels = []
test_data2 = []

class_mapping = {
    'Amigo': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Chrome': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'Maxthon': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'Opera': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'RedApp': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'Safari': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'Tor': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'Via': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'Vivaldi': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'Yandex': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # Добавьте другие классы, если необходимо
}

first_row = df.iloc[0]

rows = df.iloc[0:2]

# Определяем размер тестовой выборки
test_size = int(0.2 * len(df))

# Получаем случайные индексы для тестовой выборки
test_indices = df.sample(n=test_size, random_state=129).index

# Разделяем выборки
train_df = df.drop(test_indices)
test_df = df.loc[test_indices]
train_df = train_df.sample(frac=1, random_state=629)

print(df.info(memory_usage='deep'))


for j in range(20):
    my_data = [random.randint(0, 1) for i in range(10)]
    class_label = [0, 1, 0]
    training_data.append((my_data, class_label))

for i in range(1, len(train_df)):
    pixels = list(map(int, train_df.iloc[i]['pixels'].split(',')))
    class_label = class_mapping[train_df.iloc[i]['class']]
    # class_label = [0, 1, 0]
    training_data2.append((pixels, class_label))

for i in range(1, len(test_df)):
    pixels = list(map(int, test_df.iloc[i]['pixels'].split(',')))
    class_label = class_mapping[test_df.iloc[i]['class']]
    # class_label = [0, 1, 0]
    test_data2.append((pixels, class_label))

net = Network([4096, 128, 10])

net.SGD(training_data2, 30, 0.01)

# Сохранение весов модели

# with open('model_params_weight.pkl', 'wb') as f:
#     pickle.dump(net.weights, f)
#
# with open('model_params_biases.pkl', 'wb') as f:
#     pickle.dump(net.biases, f)

# Загрузка весов модели

# with open('model_params_weight.pkl', 'rb') as f:
#     net.weights = pickle.load(f)
#
# with open('model_params_biases.pkl', 'rb') as f:
#     net.biases = pickle.load(f)


acc = net.calc_accuracy(data=test_data2)

print(f'Accuracy = {acc}')

# import matplotlib.pyplot as plt
#
# plt.plot(loss_arr)
# plt.show()

