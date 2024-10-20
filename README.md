# Многослойный перцептрон для многоклассовой классификации логотипов браузеров

## Описание проекта
Данный проект реализует многослойный перцептрон (MLP) для решения задачи многоклассовой классификации изображений логотипов браузеров. Целью является создание нейронной сети нуля, которая предоставит возможность глубокого понимания работы многослойного перцептрона, включая архитектуру сети, алгоритм обратного распространения ошибки и использование разных функций активации. Также требуется разработать графический пользовательский интерфейс (GUI), который позволит пользователям настраивать параметры сети (количество слоёв и нейронов), визуализировать метрики, а также давать возможность протестировать работы сети.

Данные для обучения и тестирования брались из следущего репозитория: https://github.com/Maximiliano-16/DataSetCV/tree/main

## Используемые технологии
- **Python**: основной язык программирования.
- **NumPy**: для численных вычислений.
- **Pandas** для работы с датасетом
- **Matplotlib**: для визуализации графиков и метрик.
- **tkinter**: для создания графического пользовательского интерфейса (GUI).

## Архитектура сети
- **Функция активации**:
  - Sigmoid
  - ReLU (Rectified Linear Unit)
  - Softmax

- **Функция потерь**:
  - Multiclass Cross-Entropy Loss

- **Алгоритм обучения**:
  - Обратное распространение ошибки (Backpropagation)

## GUI
Проект включает графический интерфейс, который позволяет пользователю настраивать параметры нейронной сети, такие как количество слоёв и количество нейронов в каждом слое. Также есть вохможность нарисовать логотип браузере и протестировать работу нейронной сети на практике. Ниже приведены скриншоты интерфейса:

![Распознал яндекс](https://github.com/user-attachments/assets/65d63da9-96c6-4174-9d7d-53d91a3c1d0a)

![chrome 2](https://github.com/user-attachments/assets/6df164d3-cc86-48f3-bdac-26214323b836)

Во время обучения нейронной сети оценивались следующие метрики:
- **Accuracy**: точность классификации, измеряет долю правильных предсказаний по сравнению с общим количеством предсказаний.
- **Precision**: точность положительных предсказаний, показывает долю истинных положительных среди всех положительных предсказаний.
- **Recall**: полнота, показывает долю истинных положительных предсказаний среди всех фактических положительных примеров.
- **F1-score**: гармоническое среднее Precision и Recall, дающее сбалансированное представление о производительности модели.
- **Loss**: значение функции потерь.

На графиках ниже представлено изменение метрик в процессе обучения модели:

![acc](https://github.com/user-attachments/assets/2ceb0755-55f1-454d-8836-24b3f592ac3e)

![f1-score](https://github.com/user-attachments/assets/a6954792-0f87-4cde-bb38-2986d3174924)

![loss](https://github.com/user-attachments/assets/e9a776a7-e542-4d8c-aac3-5898506ff6c5)

## Заключение

В результате работы над проектом был успешно реализован многослойный перцептрон для задач многоклассовой классификации изображений логотипов веб-браузеров. Модель была обучена с использованием различных функций активации и методов оптимизации, что позволило достичь высоких показателей точности и надежности при классификации.

В ходе обучения модели были получены следующие финальные значения метрик:

- **Accuracy**: _0.994962_
- **Precision**: _0.986874_
- **Recall**: _0.985831_
- **F1-score**: _0.986901_
- **Loss**: _0.00018_

Эти результаты продемонстрировали, что наша модель эффективно распознает логотипы браузеров, обеспечивая высокую точность и баланс между точностью и полнотой. Использование различных функций активации и тщательный выбор архитектуры сети оказали значительное влияние на производительность модели.
