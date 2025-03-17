import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# Преобразование разных типов представления неточных данных к формату,
# унифицированному для дальнейшего применения Data Reconciliation.

# Класс для Dempster-Shafer (структура Демпстера-Шафера)
class DempsterShafer:
    tolerance_threshold = 0.2  # Константа для объединения близких границ

    def __init__(self, intervals, masses):
        # Проверка входных аргументов
        if intervals.shape[1] != 2:
            raise ValueError('Поле Intervals должно быть матрицей размера N×2.')
        if np.any(intervals[:, 0] > intervals[:, 1]):
            raise ValueError('В поле Interval левые границы интервалов должны быть меньше правых границ.')
        if intervals.shape[0] != len(masses):
            raise ValueError('Количество фокальных элементов должно совпадать с переданным числом масс.')
        if np.any(masses < 0) or np.any(masses > 1):
            raise ValueError('Значения в поле Masses должны быть от 0 до 1.')
        threshold = 1e-2
        if abs(np.sum(masses) - 1) > threshold:
            raise ValueError(f'Сумма масс должна быть равна 1 (текущая сумма: {np.sum(masses):.6f})')

        self.Intervals = intervals
        self.Masses = masses.reshape(-1, 1)  # Превращение в вектор-столбец

    # Сложение по правилу Демпстера (нормализированное)
    def __add__(self, other):
        if not isinstance(other, DempsterShafer):
            raise TypeError('Оба операнда должны быть объектами типа DempsterShafer.')

        # Вычисление всех возможных сумм границ интервалов
        n, m = np.meshgrid(np.arange(self.Intervals.shape[0]), np.arange(other.Intervals.shape[0]))
        n = n.flatten()
        m = m.flatten()

        # Вычисление комбинированных интервалов
        sum_intervals = self.Intervals[n] + other.Intervals[m]

        # Вычисление комбинированных масс (произведение)
        comb_masses = self.Masses[n] * other.Masses[m]

        # Во избежание комбинаторного взрыва, объединяем близкие границы
        unique_int, idx = np.unique(sum_intervals, axis=0, return_inverse=True)
        sum_masses = np.bincount(idx, weights=comb_masses.flatten())

        # Нормализация после агрегации
        if len(sum_masses) < len(comb_masses):
            sum_masses = sum_masses / np.sum(sum_masses)

        return DempsterShafer(unique_int, sum_masses)

    # Умножение на константу
    def __rmul__(self, k):
        if not isinstance(k, (int, float)):
            raise TypeError('Операция умножения задана только для перемножения с коэффициентом.')
        z_intervals = k * self.Intervals
        if k < 0:
            z_intervals = z_intervals[:, [1, 0]]
        return DempsterShafer(z_intervals, self.Masses)

    def getStd(self, method='approximate'):
        intervals = self.Intervals  # Переменная для сокращенной записи
        masses = self.Masses.flatten()  # Переменная для сокращенной записи
        n = intervals.shape[0]

        if method == 'approximate':
            stdCint = np.array([np.nan, np.nan])
            # 1. Минимальное значение среднеквадратического отклонения
            # Минимальная дисперсия внутриинтервального распределения (0) + дисперсия центров интервалов
            midpoints = np.mean(intervals, axis=1)
            mean_mid = np.dot(masses, midpoints)
            between_var = np.sum(masses * (midpoints - mean_mid) ** 2)
            stdCint[0] = np.sqrt(between_var)

            # 2. Максимальное значение дисперсии
            # Максимальная дисперсия внутриинтервального распределения + дисперсия центров интервалов
            within_var = np.sum(masses * np.diff(intervals, axis=1) ** 2 / 4)
            # Случай, когда имеет место равномерное дискретное распределение, сосредоточенное в границах фокального интервала
            stdCint[1] = np.sqrt(within_var + between_var)

        elif method == 'accurate':
            # Оптимизация
            # 1. Минимальное значение дисперсии
            H = 2 * (np.diag(masses) - np.outer(masses, masses))
            lb = intervals[:, 0]  # Левые границы интервалов
            ub = intervals[:, 1]  # Правые границы интервалов

            def objective(x):
                return np.dot(x, np.dot(H, x))

            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = [(lb[i], ub[i]) for i in range(n)]
            res = minimize(objective, np.mean(intervals, axis=1), bounds=bounds, constraints=constraints)
            mean_min = np.dot(masses, res.x)
            minVar = np.sum(masses * (res.x - mean_min) ** 2)

            # Оптимизация
            # 2. Максимальное значение дисперсии
            maxVar = self.getStd(method='approximate')[1] ** 2  # Начальное приближение

            # Определение оптимальной точки (комбинаторный путь)
            if n <= 15:  # Практическое ограничение во избежание комбинаторного взрыва
                # Генерация всех возможных граничных точек (2^n)
                combinations = np.array([list(map(int, list(bin(i)[2:].zfill(n)))) for i in range(2 ** n)])
                combinations = combinations[:, ::-1]  # Двоичный порядок

                # Конвертация в граничные точки
                x_vals = intervals[:, 0] + (intervals[:, 1] - intervals[:, 0]) * combinations

                # Вычисление дисперсии для всех комбинаций
                for i in range(combinations.shape[0]):
                    x = x_vals[i]
                    current_mean = np.dot(masses, x)
                    current_var = np.sum(masses * (x - current_mean) ** 2)
                    if current_var > maxVar:
                        maxVar = current_var
            else:
                # Проверка экстремальных комбинаций граничных точек (ускорение)
                x_left = intervals[:, 0]
                var_left = np.sum(masses * (x_left - np.mean(x_left)) ** 2)
                x_right = intervals[:, 1]
                var_right = np.sum(masses * (x_right - np.mean(x_right)) ** 2)
                maxVar = max(maxVar, var_left, var_right)

            stdCint = np.sqrt([minVar, maxVar])

        else:
            raise ValueError('Аргумент method не задан. Допустимые значения: approximate, accurate.')

        return stdCint

    # Построение графика фокальных элементов с массами
    def plot(self):
        plt.figure()

        # Сортировка интервалов для лучшей визуализации
        sorted_int = np.sort(self.Intervals, axis=0)
        order = np.argsort(self.Intervals[:, 0])
        sorted_masses = self.Masses[order]

        # Отображение каждого фокального элемента
        for i in range(sorted_int.shape[0]):
            int = sorted_int[i]
            mass = sorted_masses[i]
            plt.plot(int, [mass, mass], 'b-o', linewidth=1.5, markerfacecolor='b', markersize=4)
            # Добавление текста о массах
            plt.text(np.mean(int), mass + 0.02, f'{float(mass):.3f}', horizontalalignment='center')

        # Форматирование графика
        plt.ylim([0, np.max(self.Masses) * (1 + self.tolerance_threshold)])
        plt.xlabel('Interval')
        plt.ylabel('Mass')
        plt.title('Структура Демпстера-Шафера.')
        plt.grid(True)
        plt.show()

    # Преобразование в PBox
    def DempsterShafer2PBox(self, numPoints=None):
        if numPoints is None:
            numPoints = len(self.Masses)
        # Дискретизация носителя
        x_vals = np.linspace(np.min(self.Intervals), np.max(self.Intervals), numPoints)
        borders = np.zeros((numPoints, 2))

        for i in range(numPoints):
            x = x_vals[i]
            # Belief (нижняя граница CDF)
            borders[i, 0] = np.sum(self.Masses[self.Intervals[:, 0] <= x])
            # Plausibility (верхняя граница CDF)
            borders[numPoints - i - 1, 1] = np.sum(self.Masses[self.Intervals[:, 1] >= x])

        lowerCDF = np.min(borders, axis=1)
        upperCDF = np.max(borders, axis=1)
        return PBox(x_vals, lowerCDF, upperCDF)

    # Преобразование в Hist
    def DempsterShafer2Hist(self, numBins=None):
        if numBins is None:
            numBins = len(self.Intervals)
        # Определение границ полос гистограммы
        all_points = np.concatenate([self.Intervals[:, 0], self.Intervals[:, 1]])
        edges = np.linspace(np.min(all_points), np.max(all_points), numBins + 1)

        lowerPDF = np.zeros(numBins)
        upperPDF = np.zeros(numBins)

        for i in range(numBins):
            left = edges[i]
            right = edges[i + 1]
            # Оценка границ вероятностей
            lowerPDF[i] = np.sum(self.Masses[(self.Intervals[:, 0] >= left) & (self.Intervals[:, 1] <= right)])
            upperPDF[i] = np.sum(self.Masses[(self.Intervals[:, 0] <= right) & (self.Intervals[:, 1] >= left)])

        # Нормализация по длине полос
        bin_widths = np.diff(edges)
        lowerPDF = lowerPDF / bin_widths
        upperPDF = upperPDF / bin_widths

        return Hist(edges, lowerPDF, upperPDF)

    # Преобразование в Fuzzy
    def DempsterShafer2Fuzzy(self, numPoints=None):
        if numPoints is None:
            numPoints = len(self.Masses)
        # Дискретизация носителя
        x_vals = np.linspace(np.min(self.Intervals), np.max(self.Intervals), numPoints)
        membership = np.zeros_like(x_vals)

        # Вычисление plausibility для каждой точки
        for i in range(len(x_vals)):
            x = x_vals[i]
            membership[i] = np.sum(self.Masses[(self.Intervals[:, 0] <= x) & (self.Intervals[:, 1] >= x)])

        return Fuzzy(x_vals, membership / np.max(membership))

    # Преобразование в FuzzyInterval
    def DempsterShafer2FuzzyInterval(self):
        # Используем значения масс в качестве уровней значимости
        sortedMasses = np.sort(self.Masses, axis=0)[::-1]
        alphaLevels = np.cumsum(sortedMasses)

        # Вычисление вложенных интервалов
        sorted_indices = np.argsort(self.Masses, axis=0)[::-1]
        intervals = self.Intervals[sorted_indices]
        intervals = intervals.reshape(-1, 2)

        for i in range(1, intervals.shape[0]):
            intervals[i, 0] = np.minimum(intervals[i - 1, 0], intervals[i, 0])
            intervals[i, 1] = np.maximum(intervals[i - 1, 1], intervals[i, 1])

        return FuzzyInterval(alphaLevels[::-1], intervals)