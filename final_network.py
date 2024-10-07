import random

# Сторонние библиотеки
import numpy as np
import pandas as pd


class Network(object):

    def __init__(self, sizes):
        """ Массив sizes содержит количество нейронов в соответствующих слоях. Так что, если мы хотим создать объект Network с двумя нейронами в первом слое, тремя нейронами во втором слое, и одним нейроном в третьем, то мы запишем это, как [2, 3, 1]. Смещения и веса сети инициализируются случайным образом с использованием распределения Гаусса с математическим ожиданием 0 и среднеквадратичным отклонением 1. Предполагается, что первый слой нейронов будет входным, и поэтому у его нейронов нет смещений, поскольку они используются только при подсчёте выходных данных. """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.t = [0]*sizes[-1]


        ######## new
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """Возвращает выходные данные сети, когда ``a`` - входные данные."""
        for b, w in zip(self.biases, self.weights):
            a = ReLu(a @ w + b)
            # a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Обучаем сеть при помощи мини-пакетов и стохастического градиентного спуска. training_data – список кортежей "(x, y)", обозначающих обучающие входные данные и желаемые выходные. Остальные обязательные параметры говорят сами за себя. Если test_data задан, тогда сеть будет оцениваться относительно проверочных данных после каждой эпохи, и будет выводиться текущий прогресс. Это полезно для отслеживания прогресса, однако существенно замедляет работу. """


        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            for i in range(n):

                # random.shuffle(training_data)
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                x, y = training_data[i][0], training_data[i][1]
                # print(f'x = {x}, y = {y}')
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                self.weights = [w - (eta / len(training_data[i])) * nw
                                for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - (eta / len(training_data[i])) * nb
                               for b, nb in zip(self.biases, nabla_b)]

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


    def backprop(self, x, y):
        """Вернуть кортеж ``(nabla_b, nabla_w)``, представляющий градиент для функции стоимости C_x.  ``nabla_b`` и ``nabla_w`` - послойные списки массивов numpy, похожие на ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        # прямой проход
        activation = x
        activations = [x] # список для послойного хранения активаций
        counter = 0
        zs = [] # список для послойного хранения z-векторов
        activation = np.matrix(activation)
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = activation @ w + b
            zs.append(z)
            relu = ReLu(z)
            activation = relu
            activations.append(activation)
            counter+=1

        # # Последний слой
        b = self.biases[-1]
        w = self.weights[-1]
        z = activation @ w + b
        zs.append(z)
        activation = self.newsoftmax(z)
        activations.append(activation)
        counter += 1

        # обратный проход
        res_entropy = self.binary_cross_entropy(y, activations[-1])

        ##########################################
        # ДЛЯ ТРЕХ СЛОЕВ

        dE_dtlast = self.cost_derivative(activations[-1], y)

        dE_dW2 = activations[-2].T @ dE_dtlast

        dE_db2 = dE_dtlast
        dE_dh1 = dE_dtlast @ self.weights[-1].T

        dE_dt1 = np.array(dE_dh1) * np.array(ReLu_deriv(zs[-2]))
        dE_dW1 = np.matrix(activations[0]).T @ dE_dt1
        dE_db1 = dE_dt1

        nabla_b[0] = dE_db1
        nabla_w[0] = dE_dW1
        nabla_b[1] = dE_db2
        nabla_w[1] = dE_dW2

        return (nabla_b, nabla_w)

    def my_evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        print(test_result)


    def evaluate(self, test_data):
        """Вернуть количество проверочных входных данных, для которых нейросеть выдаёт правильный результат. Выходные данные сети – это номер нейрона в последнем слое с наивысшим уровнем активации."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Вернуть вектор частных производных (чп C_x / чп a) для выходных активаций."""
        return (output_activations-y)

    def binary_cross_entropy(self, t, p):

        t = np.float_(t)
        p = np.float_(p).T
        # binary cross-entropy loss
        return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))

    def softmax(self, z):
        """Softmax функция для последнего слоя."""
        exp_z = np.exp(z - np.max(z))  # Для стабильности
        return exp_z / exp_z.sum(axis=0)

    def newsoftmax(self, z):
        out = np.exp(z)
        return out / np.sum(out)

    def predict(self, x):
        t1 = x @ self.weights[0] + self.biases[0]
        h1 = ReLu(t1)
        t2 = h1 @ self.weights[1] + self.biases[1]
        z = self.newsoftmax(t2)
        return z

    def calc_accuracy(self, data):
        correct = 0
        for i in range(len(data)-1):
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
    return 1.0/(1.0+np.exp(-z))

def ReLu(z):
    return np.maximum(z, 0)

def ReLu_deriv(z):
    return (z >= 0).astype(float)

def sigmoid_prime(z):
    """Производная сигмоиды."""
    sig = sigmoid(z)
    print(f'sig = {sig}\n')
    print(f'type(sig) = {type(sig)}\n')
    print(f'(1-sig) = {(1-sig)}')
    return sig @ (1-sig).transpose()

df = pd.read_csv('D:/MLUniversity/work1/Dataset/BrowserLogos/output.csv')

training_data = []
training_data2 = []
class_label = []
pixels = []

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
}

first_row = df.iloc[0]

rows = df.iloc[0:2]
print(len(list(map(int, rows['pixels'][0].split(',')))))
print(df['class'][1])

for i in range(len(df)):
    pixels = list(map(int, df['pixels'][0].split(',')))
    class_label = class_mapping[df['class'][1]]
    training_data2.append((pixels, class_label))


net = Network([16384, 1024, 10])
net.SGD(training_data2, 3, 1, 0.1)
acc = net.calc_accuracy(data=training_data2)

# acc = self.calc_accuracy(data=training_data)
print(f'Accuracy = {acc}')

