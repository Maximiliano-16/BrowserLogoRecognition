import random

# Сторонние библиотеки
import numpy as np
import pandas as pd
import csv


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
            a = ReLu(a @ w + b)
            # a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Обучаем сеть при помощи мини-пакетов и стохастического градиентного спуска. training_data – список кортежей "(x, y)", обозначающих обучающие входные данные и желаемые выходные. Остальные обязательные параметры говорят сами за себя. Если test_data задан, тогда сеть будет оцениваться относительно проверочных данных после каждой эпохи, и будет выводиться текущий прогресс. Это полезно для отслеживания прогресса, однако существенно замедляет работу. """


        if test_data: n_test = len(test_data)
        n = len(training_data)
        # print(f'n = {n}\n'
              # f'mini_batch_size = {mini_batch_size}\n')
        for j in range(epochs):
            print(f'training_data = \n {training_data}')
            random.shuffle(training_data)
            print(f'training_data = \n {training_data}')
            for i in range(n):


                # random.shuffle(training_data)
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                # print(f'training_data[i] = {training_data[i][1]}')
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


                # self.update_mini_batch(training_data[i], eta)

            # mini_batches = [
            #     training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # # print(f'mini_batches = {mini_batches}')
            # for mini_batch in mini_batches:
            #     print(f'mini_batch = {mini_batch}')
            #     self.update_mini_batch(mini_batch, eta)



                # print("Epoch {0}: {1} / {2}".format(
                #     j, self.my_evaluate(training_data), len(training_data)))

    def update_mini_batch(self, mini_batch, eta):
        """Обновить веса и смещения сети, применяя градиентный спуск с использованием обратного распространения к одному мини-пакету. mini_batch – это список кортежей (x, y), а eta – скорость обучения."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # print(f'mini_batch = {mini_batch}')
        # for x, y in mini_batch:
            # print('IN UPDATE MINI BATCH')
            # print(f'x = {x}\n'
            #       f'y = {y}\n'
            #       f'------------')

            # print(f'IN update_mini_batch\n'
            #       f'nabla_b = {nabla_b}\n'
            #       f'nabla_w = {nabla_w}\n')

        # print(f'Веса и смещения ДО изменения\n'
        #       f'self.weights = {self.weights}\n'
        #       f' self.biases = { self.biases}\n')

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        # print(f'Веса и смещения ПОСЛЕ изменения\n'
        #       f'self.weights = {self.weights}\n'
        #       f' self.biases = {self.biases}\n')

    def backprop(self, x, y):
        """Вернуть кортеж ``(nabla_b, nabla_w)``, представляющий градиент для функции стоимости C_x.  ``nabla_b`` и ``nabla_w`` - послойные списки массивов numpy, похожие на ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # print(f'nabla_b = {nabla_b}\n'
        #       f'nabla_w = {nabla_w}\n')
        ALPHA = 0.01
        NUM_EPOCHS = 10



        # прямой проход
        activation = x
        activations = [x] # список для послойного хранения активаций
        # print('IN BACKPROP')
        # print(f'activation = {activation}\n'
        #       f'activations = {activations}\n')
        counter = 0
        zs = [] # список для послойного хранения z-векторов
        activation = np.matrix(activation)
        # print('start for')
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            # print('-------')
            # print(f'Итерация = {counter}\n')
            # print(f'b = {b}, w = {w}\n')
            # print(f'Вычисление z c использованием: w, activation, b')
            # print(type(np.matrix(activation)))

            # print(f'activation = {activation}\n')

            z = activation @ w + b
            # print(f'z = {z}\n')
            zs.append(z)
            # print(f'zs = {zs}\n')
            # activation = sigmoid(z)
            relu = ReLu(z)
            activation = relu
            # print(f'RELU = {relu}\n')
            # print(f'after relu(z) new activation = {activation}\n')
            activations.append(activation)
            # print(f'activations = {activations}\n')
            counter+=1

        # # Последний слой
        # print(f'Последняя итерация = {counter}\n')
        # print(f'self.biases[:-1] = {self.biases[-1]}')
        # print(f'self.weights[:-1] = {self.weights[-1]}')
        b = self.biases[-1]
        w = self.weights[-1]
        z = activation @ w + b
        # print(f'activation = {activation}\n')
        # print(f'z = {z}\n')
        zs.append(z)
        # print(f'zs = {zs}\n')
        activation = self.newsoftmax(z)
        # print(f'after softmax(z) new activation = {activation}\n')
        activations.append(activation)
        # print(f'activations = {activations}\n')
        counter += 1

        # обратный проход
        # print('--------------------\n')
        # print(f'y = {y}')
        res_entropy = self.binary_cross_entropy(y, activations[-1])
        # print(f'res_entropy = {res_entropy}\n')



        ##########################################
        # ДЛЯ ТРЕХ СЛОЕВ

        # print('--------------------\n'
        #       'Обратный проход')
        # print(f'Вычисляется delta, вызыванием функции cost_derivative с параметрами:\n'
        #       f'activations[-1] = {activations[-1]}\n'
        #       f'y = {y}, zs[-1] = {zs[-1]}')
        dE_dtlast = self.cost_derivative(activations[-1], y)
        # print(f'dE_dtlast = {dE_dtlast}')
        # print(f'activations[-2] = {activations[-2]}, dE_dtlast = {dE_dtlast}\n')
        dE_dW2 = activations[-2].T @ dE_dtlast
        # print(f'dE_dW2 = {dE_dW2}')
        dE_db2 = dE_dtlast
        # print(f'dE_db2 = {dE_db2}\n')
        # print(f'self.weights[-1].T = {self.weights[-1].T}')
        dE_dh1 = dE_dtlast @ self.weights[-1].T
        # print(f'dE_dh1 = {dE_dh1}\n')
        # print(f'dE_dh1 = {type(dE_dh1)}\n')
        # print(f'zs = {zs}\n')
        # print(f'ReLu_deriv(zs[-2]) = {ReLu_deriv(zs[-2])}')


        dE_dt1 = np.array(dE_dh1) * np.array(ReLu_deriv(zs[-2]))
        # print(f'dE_dt1 = {dE_dt1}\n')

        # print(f'np.matrix(activations[0]).T = {np.matrix(activations[0]).T}')

        dE_dW1 = np.matrix(activations[0]).T @ dE_dt1
        # print(f'dE_dW1 = {dE_dW1}')
        dE_db1 = dE_dt1

        # Update

        # self.weights[0] = self.weights[0] - ALPHA * dE_dW1
        # self.biases[0] = self.biases[0] - ALPHA * dE_db1
        # self.weights[1] = self.weights[1] - ALPHA * dE_dW2
        # self.biases[1] = self.biases[1] - ALPHA * dE_db2


        # print(f'\n'
        #       f'---------------------------\n'
        #       f'ФИНАЛЬНЫЕ ЗНАЧЕНИЯ\n'
        #       f'dE_db1 = {dE_db1}\n'
        #       f'dE_dW1 = {dE_dW1}\n'
        #       f'dE_db2 = {dE_db2}\n'
        #       f'dE_dW2 = {dE_dW2}\n')

        nabla_b[0] = dE_db1
        nabla_w[0] = dE_dW1
        nabla_b[1] = dE_db2
        nabla_w[1] = dE_dW2

        # print(f'nabla_b = {nabla_b}\n'
        #       f'nabla_w = {nabla_w}\n')





        ######################################

        # delta =  self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # print(f'dE_dtlast = {dE_dtlast}')
        # print(f'nabla_b = {nabla_b}')
        #
        # # print(f'delta = {delta}')
        # # print(f'weight = {self.weights}')
        # nabla_b[-1] = dE_dtlast
        # print(f'Вычисляется nabla_w[-1] с использованим:\n'
        #       f'delta = {dE_dtlast}, activations[-2].transpose() = {activations[-2].transpose()}\n')
        # nabla_w[-1] = np.dot(dE_dtlast, activations[-2].transpose())
        # print(f'nabla_w[-1] = {nabla_w[-1]}')
        #
        # """Переменная l в цикле ниже используется не так, как описано во второй главе книги. l = 1 означает последний слой нейронов, l = 2 – предпоследний, и так далее. Мы пользуемся преимуществом того, что в python можно использовать отрицательные индексы в массивах."""
        # for l in range(2, self.num_layers):
        #     print('В цикле по слоям')
        #     z = zs[-l]
        #     print(f'l = {l}\n'
        #           f'z(zs[-l]) = {z}\n'
        #           f'type(z) = {type(z)}')
        #     sp = sigmoid_prime(z)
        #     print(f'sp = {sp}\n')
        #     # print("Shape of delta:", delta.shape)
        #     # print("Shape of weights[-l-1]:", self.weights[-l].transpose().shape)
        #     # delta = np.dot(self.weights[-l-1].transpose(), delta) * sp
        #     delta = np.dot(delta, self.weights[-l - 1].transpose()) * sp
        #     nabla_b[-l] = delta
        #     # print(f'self.num_layers {self.num_layers}')
        #     # print(f'activations - {len(activations)}')
        #
        #     nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        #     # nabla_w[-l] = np.dot(delta, activations[-l].transpose())
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

    def predict(self, x):
        # print(f'x = {x}\n'
        #       f'self.weights[0] = {self.weights[0]}\n'
        #       f'self.biases[0] = {self.biases[0]}\n')
        t1 = x @ self.weights[0] + self.biases[0]
        # print(f't1 = {t1}')
        h1 = ReLu(t1)
        # print(f'h1 = {h1}\n'
        #       f'self.weights[1] = {self.weights[1]}\n'
        #       f'self.biases[1] = {self.biases[1]}\n')
        t2 = h1 @ self.weights[1] + self.biases[1]
        # print(f't2 = {t2}')
        z = self.newsoftmax(t2)
        # print(f'z = {z}')
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

    def save_model_csv(self, weights, biases, sizes, filename):
        try:
            with open(filename, 'w') as f:
                # Сохранение размера сети
                f.write(','.join(map(str, sizes)) + '\n')

                # Сохранение весов
                for i, w in enumerate(weights):
                    layer_weights = np.concatenate(
                        w).flatten()  # Преобразование в одномерный массив
                    f.write(','.join(map(str, layer_weights)) + '\n')

                # Сохранение смещений
                for i, b in enumerate(biases):
                    layer_biases = b.flatten()  # Преобразование в одномерный массив
                    f.write(','.join(map(str, layer_biases)) + '\n')

            print(f"Модель успешно сохранена в '{filename}'.")
        except Exception as e:
            print(f"Ошибка при сохранении модели: {e}")

    def load_model_csv(self, filename):

        # Загрузка размеров сети
        sizes = []
        self.biases = []
        self.weights = []
        with open(filename, 'r') as f:
            sizes = list(map(int, f.readline().strip().split(',')))
            print('size = ', sizes)

            weights = []
            biases = []
            layer_index = 0

            # Загрузка весов
            for _ in range(len(sizes) - 1):
                layer_weights = np.array(list(map(float, f.readline().strip().split(','))))

                # Восстанавливаем форму
                layer_weights = layer_weights.reshape(sizes[layer_index],
                                                      sizes[layer_index+1])
                print(f'layer_weights = {layer_weights}\n')
                self.weights.append(layer_weights)
                layer_index += 1

            # Загрузка смещений
            layer_index = 0
            for _ in range(len(sizes) - 1):
                layer_biases = np.array(list(map(float, f.readline().strip().split(','))))

                # Восстанавливаем форму
                # layer_biases = layer_biases.reshape(sizes[layer_index+1],
                #                                     1)  # Столбец
                layer_biases = np.array([layer_biases])
                print(f'layer_biases = {layer_biases}\n')
                self.biases.append(layer_biases)
                layer_index += 1

        print(f"Модель успешно загружена из '{filename}'.")
        self.sizes = sizes
        self.num_layers = len(sizes)



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
    # Добавьте другие классы, если необходимо
}

first_row = df.iloc[0]

rows = df.iloc[0:2]
print(len(list(map(int, rows['pixels'][0].split(',')))))
print(df['class'][1])

# Определяем размер тестовой выборки
test_size = int(0.2 * len(df))

# Получаем случайные индексы для тестовой выборки
test_indices = df.sample(n=test_size, random_state=42).index

# Разделяем выборки
train_df = df.drop(test_indices)
test_df = df.loc[test_indices]
train_df = train_df.sample(frac=1, random_state=42)

print(train_df.head())
print(test_df.head())
print(train_df.shape)
print(test_df.shape)

# for i in range(5):
#     # pixels = list(map(int, rows['pixels'][0].split(',')))[:10]
#     class_label = class_mapping[rows['class'][1]]
#     print(class_label)
# # print(rows['pixels'][0])
for i in range(1, len(df), 200):
    pixels = list(map(int, df['pixels'][0].split(',')))[:5]
    class_label = class_mapping[df['class'][1]]
    class_label = [0, 1, 0]
    training_data2.append((pixels, class_label))


  # Преобразуем строку в список целых чисел
# print(training_data2)
# pixels = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
# pixels = [0, 1, 1, 0, 0]
# class_label = class_mapping.get(first_row['class'], 0)  # Получаем метку класса, по умолчанию 0
# class_label = [0, 1, 0]
# class_label = np.matrix(class_label)
training_data.append((pixels, class_label))
print(training_data2)

net = Network([5, 4, 3])
# print(f'net.weights = {net.weights}')
# net.save_model_csv(weights=net.weights, biases=net.biases, sizes=net.sizes, filename='model_params.csv')
net.load_model_csv('model_params.csv')
# print(f'net.weights = {net.weights}\n')
# print(f'net.biases = {net.biases}')

net.SGD(training_data2, 3, 1, 0.1)
acc = net.calc_accuracy(data=training_data2)

# acc = self.calc_accuracy(data=training_data)
print(f'Accuracy = {acc}')


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

