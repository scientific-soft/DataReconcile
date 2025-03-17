import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, binom, gaussian_kde
from scipy.optimize import minimize
from sklearn.cluster import KMeans

# Преобразование разных типов представления неточных данных к формату,
# унифицированному для дальнейшего применения Data Reconciliation.

# Класс для одномерной выборки
class Sample:
    def __init__(self, x):
        # Проверка размерностей входных данных
        if len(x) == 0:
            raise ValueError('В выборке должно быть хотя бы одно значение.')
        if np.any(np.isnan(x)):
            raise ValueError('Выборка не может содержать значения типа NaN.')
        self.x = np.array(x).flatten()  # Принудительный перевод в вектор-строку

    # Оператор сложения двух выборок (операнды: Z = A + B)
    def __add__(self, other):
        if not isinstance(other, Sample):
            raise TypeError('Оба операнда должны быть объектами типа Sample.')

        # Проверка размерности выборок
        n1 = len(self.x)
        n2 = len(other.x)
        if n1 == n2:
            z_values = self.x + other.x
        elif n1 < n2:
            # Дополняем значения из первой выборки бутстрепом
            z_values = other.x.copy()
            z_values[:n1] += self.x
            z_values[n1:] += self.x[np.random.randint(0, n1, n2 - n1)]
        else:
            # Дополняем значения из второй выборки бутстрепом
            z_values = self.x.copy()
            z_values[:n2] += other.x
            z_values[n2:] += other.x[np.random.randint(0, n2, n1 - n2)]

        return Sample(z_values)

    # Умножение на коэффициент
    def __rmul__(self, k):
        if not isinstance(k, (int, float)):
            raise TypeError('Операция умножения задана только для перемножения с коэффициентом.')
        return Sample(k * self.x)

    # Определение величины возможного среднеквадратического отклонения
    def getStd(self, method='approximate'):
        n = len(self.x)
        if method == 'approximate':
            x = np.sort(self.x)
            # Согласно теореме Крейновича-Тао
            q = [0.05, 0.95]  # Искомые квантили
            left = np.zeros(2)
            right = np.zeros(2)
            for i in range(2):
                k1 = max(int(np.floor(n * q[i] - np.sqrt(n * q[i] * (1 - q[i])) * norm.ppf(0.975))), 1)
                k2 = min(int(np.ceil(n * q[i] + np.sqrt(n * q[i] * (1 - q[i])) * norm.ppf(0.975))), n)
                left[i] = x[k1 - 1]
                right[i] = x[k2 - 1]
            stdCint = 0.25 * np.array([left[1] - right[0], right[1] - left[0]])
        elif method == 'accurate':
            s = np.std(self.x, ddof=1)  # Несмещенная оценка стандартного отклонения
            if n < 10:  # Используем бутстреп
                M = 100  # Количество повторений
                buf = np.zeros(M)
                for i in range(M):
                    # Генерация бутстреп-выборки с заменой
                    sample = self.x[np.random.randint(0, n, n)]
                    # Вычисление дисперсии для текущей выборки
                    buf[i] = np.var(sample, ddof=1)  # Несмещенная дисперсия
                # Оценка доверительного интервала для значений дисперсии
                stdCint = np.sqrt(np.quantile(buf, [0.05, 0.95]))
            else:
                # Асимптотический доверительный интервал
                stdCint = s * np.sqrt(n - 1) / np.sqrt([chi2.ppf(0.975, n - 1), chi2.ppf(0.025, n - 1)])
        else:
            raise ValueError('Аргумент method не задан. Допустимые значения: approximate, accurate.')

        return stdCint

    # Построение графика ecdf
    def plot(self):
        plt.figure()
        x = np.sort(self.x)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where='post', color='b', linewidth=1.5)
        plt.xlabel('x')
        plt.ylabel('CDF')
        plt.legend(['Эмпирическая функция распределения'], loc='lower right')
        plt.title('Выборочная функция распределения')
        plt.grid(True)
        plt.show()

    # Преобразование в PBox
    def Sample2PBox(self):
        x = np.sort(self.x)
        n = len(x)
        D = np.sqrt(-np.log(0.025)/(2*n))-1/(6*n)
        Flo = np.clip((np.arange(0, n) / n) - D, 0, 1)
        Fup = np.clip((np.arange(1, n + 1) / n) + D, 0, 1)
        return PBox(x, Flo, Fup)

    # Преобразование в Hist
    def Sample2Hist(self, binsNum=None):
        n = len(self.x)
        if binsNum is None:
            binsNum = int(np.ceil(1 + 1.59 * np.log(n)))
        counts, edges = np.histogram(self.x, bins=binsNum)
        cints = np.array([binom.interval(0.95, n, p) for p in counts / n])
        lowerPDF = cints[:, 0] / np.diff(edges)
        upperPDF = cints[:, 1] / np.diff(edges)
        return Hist(edges, lowerPDF, upperPDF)

    # Преобразование в DempsterShafer
    def Sample2DempsterShafer(self, numFocal=None):
        n = len(self.x)
        if numFocal is None:
            numFocal = int(np.ceil(1 + 1.59 * np.log(n)))
        # Кластеризация выборки в фокальные элементы
        kmeans = KMeans(n_clusters=numFocal)
        kmeans.fit(self.x.reshape(-1, 1))
        centers = np.sort(kmeans.cluster_centers_.flatten())

        # Определение интервалов вокруг центров кластеров
        ranges = np.diff(centers) / 2
        ranges = np.concatenate([[ranges[0]], ranges, [ranges[-1]]])

        intervals = np.zeros((numFocal, 2))
        for i in range(numFocal):
            intervals[i] = [centers[i] - ranges[i], centers[i] + ranges[i + 1]]

        # Вычисление масс по числу точек, попавших в фокальные элементы
        counts, _ = np.histogram(self.x, bins=np.concatenate([intervals[:, 0], np.array([intervals[-1, 1]])]))
        masses = counts / n

        return DempsterShafer(intervals, masses / np.sum(masses))

    # Преобразование в Fuzzy
    def Sample2Fuzzy(self, numPoints=None):
        n = len(self.x)
        if numPoints is None:
            numPoints = int(np.ceil(1 + 1.59 * np.log(n)))
        # Оценка PDF через KDE
        kde = gaussian_kde(self.x)
        x_vals = np.linspace(np.min(self.x), np.max(self.x), numPoints)
        pdf_est = kde(x_vals)
        # Преобразование PDF в функцию принадлежности
        membership = pdf_est / np.max(pdf_est)
        return Fuzzy(x_vals, membership)

    # Преобразование в FuzzyInterval
    def Sample2FuzzyInterval(self, numAlpha=None):
        if numAlpha is None:
            numAlpha = 5
        alphaLevels = np.linspace(0, 1, numAlpha)
        alphaLevels = alphaLevels[::-1]
        intervals = np.zeros((len(alphaLevels), 2))

        for i, alpha in enumerate(alphaLevels):
            p = [(1 - alpha) / 2, (1 + alpha) / 2]
            intervals[i] = np.quantile(self.x, p)

        return FuzzyInterval(alphaLevels, intervals[::-1])