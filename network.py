import random
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
dt = np.dtype(np.float64)

np.seterr(divide = 'ignore')

class Network(object):

    def __init__(self, sizes):
        """ Массив sizes содержит количество нейронов в соответствующих слоях. Так что, если мы хотим создать объект Network с двумя нейронами в первом слое, тремя нейронами во втором слое, и одним нейроном в третьем, то мы запишем это, как [2, 3, 1]. Смещения и веса сети инициализируются случайным образом с использованием распределения Гаусса с математическим ожиданием 0 и среднеквадратичным отклонением 1. Предполагается, что первый слой нейронов будет входным, и поэтому у его нейронов нет смещений, поскольку они используются только при подсчёте выходных данных. """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y, )
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.t = [0]*sizes[-1]
        self.res_entropy = 100

        self.confusion_matrix = np.zeros((10, 10))
        self.y_true = []
        self.y_pred = []
        self.entropy_counter = 0

        # print('INIT PARAMS')
        # print(f'self.num_layers = {self.num_layers},\n'
        #       f'self.sizes = {self.sizes}\n'
        #       f'self.biases = {self.biases}\n'
        #       f'self.weights = {self.weights}\n'
        #       f'type self.biases = {type(self.biases[0])}\n'
        #       f'type self.weights = {type(self.weights[0])}\n'
        #
        #       f'---------------------')


    def SGD(self, training_data, epochs, eta, test_data=None):

        if test_data: n_test = len(test_data)
        n = len(training_data)
        recall_arr, accuracy_arr, precision_arr, loss_arr, f1_arr = [], [], [], [], []

        for j in range(epochs):
            random.shuffle(training_data)
            for i in range(n):
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

                x, y = training_data[i][0], training_data[i][1]
                delta_nabla_b, delta_nabla_w = self.propagation(x, y)

                counter = 0
                for w, dw in zip(self.weights, delta_nabla_w):
                    self.weights[counter] = w - eta * dw
                    counter +=1

                counter = 0
                for b, db in zip(self.biases, delta_nabla_b):
                    self.biases[counter] = b - eta * db
                    counter += 1

            acc = self.calc_accuracy(data=test_data2)
            accuracy_arr.append(acc)

            recall = self.calculate_recall()
            recall_arr.append(recall)

            precision = self.calculate_precision()
            precision_arr.append(precision)

            loss = self.multiclass_cross_entropy_loss()
            loss_arr.append(loss)

            f1_score = self.calculate_f1_score(precision, recall)
            f1_arr.append(f1_score)

            self.y_true = []
            self.y_pred = []

            print("Epoch {0} complete".format(j))
            print(f"Accuracy = {accuracy_arr}")
            print(f"Recall = {recall_arr}")
            print(f"Precision = {precision_arr}")
            print(f"Loss = {loss_arr}")
            print(f"F1 = {f1_arr}")


        self.create_graphics(accuracy_arr, 'Accuracy')
        self.create_graphics(recall_arr, 'Recall')
        self.create_graphics(recall_arr, 'Precision')
        self.create_graphics(loss_arr, 'Loss')
        self.create_graphics(f1_arr, 'F1-score')


    def propagation(self, x, y):
        """Вернуть кортеж ``(nabla_b, nabla_w)``, представляющий градиент для функции стоимости C_x.  ``nabla_b`` и ``nabla_w`` - послойные списки массивов numpy, похожие на ``self.biases`` and ``self.weights``."""

        nabla_b = [np.zeros(b.shape, dtype=dt) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=dt) for w in self.weights]

        # прямой проход
        activation = x
        activations = [x] # список для послойного хранения активаций
        counter = 0
        zs = [] # список для послойного хранения z-векторов
        activation = np.matrix(activation)

        for b, w in zip(self.biases[:-1], self.weights[:-1]):

            z = activation @ w + b
            zs.append(z)

            # relu = ReLu(z)
            sig = sigmoid(z)

            activation = sig
            activations.append(activation)
            counter+=1

        # Последний слой
        b = self.biases[-1]
        w = self.weights[-1]
        z = activation @ w + b
        zs.append(z)
        activation = self.softmax(z)
        activations.append(activation)
        counter += 1

        last_act = np.array(activations[-1][0])
        self.y_pred.append(last_act)
        self.y_true.append(y)

        entropy = self.sparse_cross_entropy_loss(last_act, y)

        # Вывод лог функции потерь
        self.entropy_counter +=1
        if self.entropy_counter == 100:
            print(f'res_entropy = {entropy}')
            self.entropy_counter = 0

        # обратный проход

        last_layer_index = self.num_layers - 1
        dE_dt = self.cost_derivative(activations[last_layer_index], y)

        for layer_index in range(last_layer_index-1, 0, -1):
            de_dW = activations[layer_index].T @ dE_dt
            de_db = dE_dt
            de_dactivation = dE_dt @ self.weights[layer_index].T
            dE_dt = np.array(de_dactivation) * np.array(
                sigmoid_deriv(zs[layer_index - 1]))

            nabla_b[layer_index] = de_db
            nabla_w[layer_index] = de_dW

        de_dW = np.matrix(x).T @ dE_dt
        de_db = dE_dt
        nabla_b[0] = de_db
        nabla_w[0] = de_dW

        return (nabla_b, nabla_w)


    def cost_derivative(self, output_activations, y):
        """Вернуть вектор частных производных (чп C_x / чп a) для выходных активаций."""
        return (output_activations-y)


    def multiclass_cross_entropy_loss(self):
        """
        Вычисляет Multiclass Cross-Entropy Loss.
        """
        # Убедимся, что вероятности нормализованы
        self.y_pred = np.clip(self.y_pred, 1e-15, 1 - 1e-15)  # избегаем логарифма 0
        loss = -np.mean(np.sum(self.y_true * np.log(self.y_pred), axis=1))
        return loss

    def sparse_cross_entropy_loss(self, p_pred, y_true):
        y_idx = np.argmax(y_true)
        return np.log(p_pred[y_idx]) * (-1)


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
        num_prop_layers = self.num_layers -1
        h = x
        for num_layer in range(0, num_prop_layers):
            t = h @ self.weights[num_layer] + self.biases[num_layer]
            if num_layer != num_prop_layers - 1:
                # h = ReLu(t)
                h = sigmoid(t)
            else:
                h = self.softmax(t)
        return h

    def calc_accuracy(self, data):
        correct = 0
        for i in range(len(data)-1):
            x, y = data[i][0], data[i][1]
            z = self.predict(x)
            y_pred = np.argmax(z)
            res = np.argmax(y)
            self.confusion_matrix[res, y_pred] +=1
            if y_pred == res:
                correct += 1

        print(f'correct = {correct}\n'
              f'len(data) = {len(data)}')
        acc = correct / (len(data)-1)
        return acc

    def load_params(self, file_weights, file_biases):
        with open(file_weights, 'rb') as f:
            self.weights = pickle.load(f)

        with open(file_biases, 'rb') as f:
            self.biases = pickle.load(f)

    def save_params(self, file_weights, file_biases):
        with open(file_weights, 'wb') as f:
            pickle.dump(self.weights, f)

        with open(file_biases, 'wb') as f:
            pickle.dump(self.biases, f)


    def res_interpreter(self, y_pred):
        res_dict = {
            0: 'Amigo',
            1: 'Chrome',
            2: 'Maxthon',
            3: 'Opera',
            4: 'RedApp',
            5: 'Safari',
            6: 'Tor',
            7: 'Via',
            8: 'Vivaldi',
            9: 'Yandex',
        }
        return res_dict[y_pred]

    def create_training_data(self, train_df):
        class_label = []
        pixels = []
        for i in range(1, len(train_df)):
            pixels = list(map(int, train_df.iloc[i]['pixels'].split(',')))
            class_label = class_mapping[train_df.iloc[i]['class']]
            # class_label = [0, 1, 0]
            training_data2.append((pixels, class_label))

        return training_data2

    def create_testing_data(self, test_df):
        class_label = []
        pixels = []
        for i in range(1, len(test_df)):
            pixels = list(map(int, test_df.iloc[i]['pixels'].split(',')))
            class_label = class_mapping[test_df.iloc[i]['class']]
            # class_label = [0, 1, 0]
            test_data2.append((pixels, class_label))

        return test_data2

    def calculate_recall(self):
        diagonal = np.diag(self.confusion_matrix)
        FN = []
        # TP = np.sum(diagonal)
        for i in range(0, 10):
            sum_of_row = np.sum(self.confusion_matrix[i])
            FN.append(sum_of_row-diagonal[i])

        print(FN)
        print(np.sum(FN))
        return np.sum(diagonal) / (np.sum(FN) + np.sum(diagonal))

    def calculate_precision(self):
        diagonal = np.diag(self.confusion_matrix)
        FP = []
        transponsed_confusion_matrix = self.confusion_matrix.T
        # TP = np.sum(diagonal)
        for j in range(0, 10):
            sum_of_row = np.sum(transponsed_confusion_matrix[j])
            FP.append(sum_of_row-diagonal[j])

        print(FP)
        print(np.sum(FP))
        return np.sum(diagonal) / (np.sum(FP) + np.sum(diagonal))

    def calculate_f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall)


    def create_graphics(self, arr, name):
        # Создание осей X (например, количество эпох)
        epochs = list(range(1, len(arr) + 1))

        # Построение графика
        plt.plot(epochs, arr, marker='o', linestyle='-', color='b')

        # Подписи осей
        plt.xlabel('Эпохи')
        plt.ylabel(name)

        # Заголовок графика
        plt.title(f'График {name}')

        # Показать сетку
        plt.grid()

        # Показать график
        plt.show()



#### Разные функции
def sigmoid(z):
    """Сигмоида."""
    return 1.0/(1.0+np.exp(-z))

def ReLu(z):
    return np.maximum(z, 0)

def ReLu_deriv(z):
    return (z >= 0).astype(float)

def deriv_sigmoid(x):
  # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)

  return fx * (1 - fx)

def sigmoid_deriv(z):
    """Производная сигмоиды. f'(x) = f(x) * (1 - f(x))"""

    sig = sigmoid(z)
    # print(f'sig = {sig}\n')
    # print(f'type(sig) = {type(sig)}\n')
    # print(f'(1-sig) = {(1-sig)}')
    return sig @ (1-sig).transpose()


df = pd.read_csv('D:/MLUniversity/work1/Dataset/BrowserLogos_64/final_output_64.csv')


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
}

# Определяем размер тестовой выборки
test_size = int(0.2 * len(df))

# Получаем случайные индексы для тестовой выборки
test_indices = df.sample(n=test_size).index

# Разделяем выборки
train_df = df.drop(test_indices)
test_df = df.loc[test_indices]
train_df = train_df.sample(frac=1)

print(train_df.head())
print(test_df.head())
print(train_df.shape)
print(test_df.shape)
print(df.info(memory_usage='deep'))

for j in range(10):
    my_data = [random.randint(0, 1) for i in range(10)]
    if j < 5:
        class_label = [0, 1, 0]
    else: class_label = [0, 0, 1]
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

if __name__ == '__main__':

    net = Network([4096, 128, 32, 10])

    net.SGD(training_data2, 10, 0.008)
    # net.save_params('model_params_weight2.pkl', 'model_params_biases2.pkl')
    # net.load_params('model_params_weight.pkl', 'model_params_biases.pkl')
    # net.save_model_csv(weights=net.weights, biases=net.biases, sizes=net.sizes, filename='model_params2.csv')
    acc = net.calc_accuracy(data=test_data2)

    print(f'Confusion matrix: \n{net.confusion_matrix}')
    print(f'RECALL: \n{net.calculate_recall()}')
    print(f'Accuracy = {acc}')

