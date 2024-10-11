import random
import sys
import pickle

# Сторонние библиотеки
import numpy as np
import pandas as pd
import csv
dt = np.dtype(np.float64)
loss_arr = []
np.seterr(divide = 'ignore')

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
        self.weights = [np.random.randn(x, y,)
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
        # print(f'n = {n}\n'
              # f'mini_batch_size = {mini_batch_size}\n')
        for j in range(epochs):
            # print(f'training_data = \n {training_data}')
            random.shuffle(training_data)
            # print(f'training_data = \n {training_data}')
            for i in range(n):


                # random.shuffle(training_data)
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                # print(f'training_data[i][] = {training_data[i][1]}'
                # print(f'training_data[0] = {training_data[0]}')
                x, y = training_data[i][0], training_data[i][1]
                # print(f'x = {x}, y = {y}')
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                # print(f'delta_nabla_b = {delta_nabla_b}\n'
                #       f'delta_nabla_w = {delta_nabla_w}\n')
                #
                # print(f'self.weight[0] = {self.weights[0]}')
                # print(f'delta_nabla_w[0] = {delta_nabla_w[0]}')

                counter = 0
                for w, dw in zip(self.weights, delta_nabla_w):
                    self.weights[counter] = w - eta * dw
                    counter +=1

                counter = 0
                for b, db in zip(self.biases, delta_nabla_b):
                    self.biases[counter] = b - eta * db
                    counter += 1

                # nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                # nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                # self.weights = [w - (eta / len(training_data[i])) * nw for w, nw in zip(self.weights, nabla_w)]
                # self.biases = [b - (eta / len(training_data[i])) * nb
                #                for b, nb in zip(self.biases, nabla_b)]

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
        nabla_b = [np.zeros(b.shape, dtype=dt) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=dt) for w in self.weights]
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
        activation = self.softmax(z)
        # print(f'after softmax(z) new activation = {activation}\n')
        activations.append(activation)
        # print(f'activations = {activations}\n')
        counter += 1

        # обратный проход
        # print('--------------------\n')
        # print(f'y = {y}')
        res_entropy = self.binary_cross_entropy(y, activations[-1])
        # print(f'y = {y}\n'
        #       f'activations[-1] = {activations[-1]}')
        # print(f'res_entropy = {res_entropy}\n')



        ##########################################
        # ДЛЯ ТРЕХ СЛОЕВ

        # print('--------------------\n'
        #       'Обратный проход')
        # print(f'Вычисляется delta, вызыванием функции cost_derivative с параметрами:\n'
        #       f'activations[-1] = {activations[-1]}\n'
        #       f'y = {y}, zs[-1] = {zs[-1]}')

        dE_dtlast = self.cost_derivative(activations[-1], y)
        dE_dt1 = dE_dtlast
        # print(f'zs = {zs}')
        for i in range(2, len(activations)):
            dE_dW2 = activations[-i].T @ dE_dtlast
            dE_db2 = dE_dt1
            dE_dh1 = dE_dt1 @ self.weights[-i+1].T
            # print(f'ReLu_deriv = {ReLu_deriv(zs[-i])}')
            dE_dt1 = np.array(dE_dh1, dtype=dt) * np.array(ReLu_deriv(zs[-i]), dtype=dt)
            nabla_b[-i+1] = dE_db2
            nabla_w[-i+i] = dE_dW2

        dE_dW1 = np.matrix(activations[0], dtype=dt).T @ dE_dt1
        dE_db1 = dE_dt1
        nabla_b[0] = dE_db1
        nabla_w[0] = dE_dW1

        loss_arr.append(res_entropy)



        # # print(f'dE_dtlast = {dE_dtlast}')
        # # print(f'activations[-2] = {activations[-2]}, dE_dtlast = {dE_dtlast}\n')
        # dE_dW2 = activations[-2].T @ dE_dtlast
        # # print(f'dE_dW2 = {dE_dW2}')
        # dE_db2 = dE_dtlast
        # # print(f'dE_db2 = {dE_db2}\n')
        # # print(f'self.weights[-1].T = {self.weights[-1].T}')
        # dE_dh1 = dE_dtlast @ self.weights[-1].T
        # # print(f'dE_dh1 = {dE_dh1}\n')
        # # print(f'dE_dh1 = {type(dE_dh1)}\n')
        # # print(f'zs = {zs}\n')
        # # print(f'ReLu_deriv(zs[-2]) = {ReLu_deriv(zs[-2])}')
        #
        #
        # dE_dt1 = np.array(dE_dh1) * np.array(ReLu_deriv(zs[-2]))
        # # print(f'dE_dt1 = {dE_dt1}\n')
        #
        # # print(f'np.matrix(activations[0]).T = {np.matrix(activations[0]).T}')
        #
        # dE_dW1 = np.matrix(activations[0]).T @ dE_dt1
        # # print(f'dE_dW1 = {dE_dW1}')
        # dE_db1 = dE_dt1
        #
        # # Update
        #
        # # self.weights[0] = self.weights[0] - ALPHA * dE_dW1
        # # self.biases[0] = self.biases[0] - ALPHA * dE_db1
        # # self.weights[1] = self.weights[1] - ALPHA * dE_dW2
        # # self.biases[1] = self.biases[1] - ALPHA * dE_db2
        #
        #
        # # print(f'\n'
        # #       f'---------------------------\n'
        # #       f'ФИНАЛЬНЫЕ ЗНАЧЕНИЯ\n'
        # #       f'dE_db1 = {dE_db1}\n'
        # #       f'dE_dW1 = {dE_dW1}\n'
        # #       f'dE_db2 = {dE_db2}\n'
        # #       f'dE_dW2 = {dE_dW2}\n')
        #
        # nabla_b[0] = dE_db1
        # nabla_w[0] = dE_dW1
        # nabla_b[1] = dE_db2
        # nabla_w[1] = dE_dW2

        # print(f'nabla_b = {nabla_b}\n'
        #       f'nabla_w = {nabla_w}\n')





        ######################################
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

    # def softmax(self, z):
    #     """Softmax функция для последнего слоя."""
    #     exp_z = np.exp(z - np.max(z))  # Для стабильности
    #
    #     return exp_z / exp_z.sum(axis=0)

    # def min_max_scaler(self, data):
    #     # Преобразуем данные в двумерный массив, если они одномерные
    #     data = np.array(data)
    #     if data.ndim == 1:
    #         data = data.reshape(1, -1)
    #
    #     # Находим минимум и максимум в каждом столбце
    #     min_vals = np.min(data, axis=1, keepdims=True)
    #     max_vals = np.max(data, axis=1, keepdims=True)
    #
    #     # Масштабируем данные
    #     scaled_data = 2 * ((data - min_vals) / (max_vals - min_vals)) - 1
    #     return scaled_data

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
        """Compute softmax values for each sets of scores in x."""
        # f = np.exp(x - np.max(x))  # shift values
        # return f / f.sum(axis=0)
        # lambda num: 0 if num < -20 else num
        def convert(num):
            return -20 if num < -20 else num
        # print(f'input softmax = {x}')
        scaled_data = self.min_max_scaler(x)
        # print(f'input scaled_data = {scaled_data}')
        # max_x = np.max(x)
        # # print(f'max_x = {max_x}')
        #
        # new_x = x - max_x
        # # print(f'new_x = {np.array(new_x)}')
        # # final_x = [0 if num < -20 else num for num in new_x[0]]
        # convertfunc = np.vectorize(convert)
        # final_x = convertfunc(new_x)
        # print(f'final_x = {final_x}')
        out = np.exp(scaled_data)
        return out / np.sum(out)


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
        z = self.softmax(t2)
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
                print(len(f.readline()))
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


df = pd.read_csv('D:/MLUniversity/work1/Dataset/BrowserLogos_64/final_output_64.csv')
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
# print(len(list(map(int, rows['pixels'][0].split(',')))))
# print(df['class'][1])

# Определяем размер тестовой выборки
test_size = int(0.2 * len(df))

# Получаем случайные индексы для тестовой выборки
test_indices = df.sample(n=test_size, random_state=129).index

# Разделяем выборки
train_df = df.drop(test_indices)
test_df = df.loc[test_indices]
train_df = train_df.sample(frac=1, random_state=629)

print(train_df.head())
print(test_df.head())
print(train_df.shape)
print(test_df.shape)
print(df.info(memory_usage='deep'))

# for i in range(5):
#     # pixels = list(map(int, rows['pixels'][0].split(',')))[:10]
#     class_label = class_mapping[rows['class'][1]]
#     print(class_label)
# # print(rows['pixels'][0])
# [np.random.randn(y, 1) for y in sizes[1:]]
# [np.random.randn(y, 1) for y in sizes[1:]]
for j in range(20):
    my_data = [random.randint(0, 1) for i in range(10)]
    class_label = [0, 1, 0]
    training_data.append((my_data, class_label))
# print(f'training_data = {training_data}')

# print('class_mapping[train_df["class"][i]]', class_mapping[train_df.iloc[0]['class']])

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


  # Преобразуем строку в список целых чисел
# print(training_data2)
# pixels = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
# pixels = [0, 1, 1, 0, 0]
# class_label = class_mapping.get(first_row['class'], 0)  # Получаем метку класса, по умолчанию 0
# class_label = [0, 1, 0]
# class_label = np.matrix(class_label)
# training_data.append((pixels, class_label))
# print(training_data2)
# etwork([16384, 1024, 10])
net = Network([4096, 128, 10])
# net = Network([10, 8, 5, 3])
# print(f'net.weights = {net.weights}')



# net.load_model_csv('model_params.csv')
# print(f'net.weights = {net.weights}\n')
# print(f'net.biases = {net.biases}')

# net.SGD(training_data2, 30, 0.01)
# net.save_model_csv(weights=net.weights, biases=net.biases, sizes=net.sizes, filename='model_params2.csv')
# with open('model_params_weight.pkl', 'wb') as f:
#     pickle.dump(net.weights, f)
#
# with open('model_params_biases.pkl', 'wb') as f:
#     pickle.dump(net.biases, f)

with open('model_params_weight.pkl', 'rb') as f:
    net.weights = pickle.load(f)

with open('model_params_biases.pkl', 'rb') as f:
    net.biases = pickle.load(f)


# f'self.biases = {net.biases[0]}\n'
# f'self.weights = {net.weights[0]}\n'

acc = net.calc_accuracy(data=test_data2)

# acc = self.calc_accuracy(data=training_data)
print(f'Accuracy = {acc}')

import matplotlib.pyplot as plt
plt.plot(loss_arr)
plt.show()






