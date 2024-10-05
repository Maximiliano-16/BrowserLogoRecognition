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

        print('INIT PARAMS')
        print(f'self.num_layers = {self.num_layers},\n'
              f'self.sizes = {self.sizes}\n'
              f'self.biases = {self.biases}\n'
              f'self.weights = {self.weights}\n'
              f'---------------------')


    def feedforward(self, a):
        """Возвращает выходные данные сети, когда ``a`` - входные данные."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Обучаем сеть при помощи мини-пакетов и стохастического градиентного спуска. training_data – список кортежей "(x, y)", обозначающих обучающие входные данные и желаемые выходные. Остальные обязательные параметры говорят сами за себя. Если test_data задан, тогда сеть будет оцениваться относительно проверочных данных после каждой эпохи, и будет выводиться текущий прогресс. Это полезно для отслеживания прогресса, однако существенно замедляет работу. """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print( "Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Обновить веса и смещения сети, применяя градиентный спуск с использованием обратного распространения к одному мини-пакету. mini_batch – это список кортежей (x, y), а eta – скорость обучения."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            print('IN UPDATE MINI BATCH')
            print(f'x = {x}\n'
                  f'y = {y}\n'
                  f'------------')
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Вернуть кортеж ``(nabla_b, nabla_w)``, представляющий градиент для функции стоимости C_x.  ``nabla_b`` и ``nabla_w`` - послойные списки массивов numpy, похожие на ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # прямой проход
        activation = x
        activations = [x] # список для послойного хранения активаций
        print('IN BACKPROP')
        print(f'activation = {activation}\n'
              f'activations = {activations}\n')
        counter = 0
        zs = [] # список для послойного хранения z-векторов
        activation = np.matrix(activation)
        print('start for')
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            print('-------')
            print(f'Итерация = {counter}\n')
            print(f'b = {b}, w = {w}\n')
            print(f'Вычисление z c использованием: w, activation, b')
            # print(type(np.matrix(activation)))

            print(f'activation = {activation}\n')
            # x_test = np.random.randn(1, 4)
            # w_test = np.random.randn(4, 5)
            # b_test = np.random.randn(1, 5)
            # print(f'x_test = {x_test} \n'
            #       f'w_test = {w_test}\n'
            #       f'b_test = {b_test}\n')
            # t_test = x_test @ w_test + b_test
            # print(f't_test = {t_test}')
            z = activation @ w + b
            print(f'z = {z}\n')
            zs.append(z)
            print(f'zs = {zs}\n')
            # activation = sigmoid(z)
            relu = ReLu(z)
            activation = relu
            print(f'RELU = {relu}\n')
            print(f'after relu(z) new activation = {activation}\n')
            activations.append(activation)
            print(f'activations = {activations}\n')
            counter+=1

        # Последний слой
        print(f'Последняя итерация = {counter}\n')
        print(f'self.biases[:-1] = {self.biases[-1]}')
        print(f'self.weights[:-1] = {self.weights[-1]}')
        b = self.biases[-1]
        w = self.weights[-1]
        z = activation @ w + b
        print(f'z = {z}\n')
        zs.append(z)
        print(f'zs = {zs}\n')
        activation = self.newsoftmax(z)
        print(f'after softmax(z) new activation = {activation}\n')
        activations.append(activation)
        print(f'activations = {activations}\n')
        counter += 1

        # обратный проход
        print('--------------------\n')
        print(f'y = {y}')
        res_entropy = self.binary_cross_entropy(y, activations[-1])
        print(f'res_entropy = {res_entropy}\n')



        ##########################################
        # ДЛЯ ТРЕХ СЛОЕВ

        print('--------------------\n'
              'Обратный проход')
        print(f'Вычисляется delta, вызыванием функции cost_derivative с параметрами:\n'
              f'activations[-1] = {activations[-1]}\n'
              f'y = {y}, zs[-1] = {zs[-1]}')
        dE_dtlast = self.cost_derivative(activations[-1], y)
        print(f'dE_dtlast = {dE_dtlast}')
        print(f'activations[-2] = {activations[-2]}, dE_dtlast = {dE_dtlast}\n')
        dE_dW2 = activations[-2].T @ dE_dtlast
        print(f'dE_dW2 = {dE_dW2}')
        dE_db2 = dE_dtlast
        print(f'dE_db2 = {dE_db2}\n')
        # dE_dh1 = dE_dtlast @



        ######################################

        # delta =  self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        print(f'dE_dtlast = {dE_dtlast}')
        print(f'nabla_b = {nabla_b}')

        # print(f'delta = {delta}')
        # print(f'weight = {self.weights}')
        nabla_b[-1] = dE_dtlast
        print(f'Вычисляется nabla_w[-1] с использованим:\n'
              f'delta = {dE_dtlast}, activations[-2].transpose() = {activations[-2].transpose()}\n')
        nabla_w[-1] = np.dot(dE_dtlast, activations[-2].transpose())
        print(f'nabla_w[-1] = {nabla_w[-1]}')

        """Переменная l в цикле ниже используется не так, как описано во второй главе книги. l = 1 означает последний слой нейронов, l = 2 – предпоследний, и так далее. Мы пользуемся преимуществом того, что в python можно использовать отрицательные индексы в массивах."""
        for l in range(2, self.num_layers):
            print('В цикле по слоям')
            z = zs[-l]
            print(f'l = {l}\n'
                  f'z(zs[-l]) = {z}\n'
                  f'type(z) = {type(z)}')
            sp = sigmoid_prime(z)
            print(f'sp = {sp}\n')
            # print("Shape of delta:", delta.shape)
            # print("Shape of weights[-l-1]:", self.weights[-l].transpose().shape)
            # delta = np.dot(self.weights[-l-1].transpose(), delta) * sp
            delta = np.dot(delta, self.weights[-l - 1].transpose()) * sp
            nabla_b[-l] = delta
            # print(f'self.num_layers {self.num_layers}')
            # print(f'activations - {len(activations)}')

            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            # nabla_w[-l] = np.dot(delta, activations[-l].transpose())
        return (nabla_b, nabla_w)

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
        # print(f't = {t}\n'
        #       f'p = {p}')
        # binary cross-entropy loss
        return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))

    def softmax(self, z):
        """Softmax функция для последнего слоя."""
        exp_z = np.exp(z - np.max(z))  # Для стабильности
        return exp_z / exp_z.sum(axis=0)

    def newsoftmax(self, z):
        out = np.exp(z)
        return out / np.sum(out)

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

class_mapping = {
    'Amigo': 1,
    # Добавьте другие классы, если необходимо
}

first_row = df.iloc[0]

pixels = list(map(int, first_row['pixels'].split(',')))[:3]  # Преобразуем строку в список целых чисел
# class_label = class_mapping.get(first_row['class'], 0)  # Получаем метку класса, по умолчанию 0
class_label = [0, 1]
# class_label = np.matrix(class_label)
training_data.append((pixels, class_label))
# print(training_data)

net = Network([3, 4, 2])
net.SGD(training_data, 2, 1, 1.0)


# # Список для хранения кортежей
# training_data = []
#
# # Определите соответствие классов (можете изменить по необходимости)
# class_mapping = {
#     'Amigo': 1,
#     # Добавьте другие классы, если необходимо
# }
#
# # Обрабатываем каждую строку
# for index, row in df.iterrows():
#     # Извлекаем пиксели и преобразуем их в список
#     pixels = list(map(int, row['pixels'].split(',')))  # Преобразуем строку в список целых чисел
#     class_label = class_mapping.get(row['class'], 0)  # Получаем метку класса, по умолчанию 0
#
#     # Добавляем в training_data
#     training_data.append((pixels, class_label))
#
# # Пример вывода первых двух кортежей
# print(training_data[:2])

# Получение первой строки


# net = Network([784, 30, 10])

