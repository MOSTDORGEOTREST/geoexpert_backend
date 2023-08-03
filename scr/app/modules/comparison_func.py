import numpy as np

def cut(x, y, x_min, x_max) -> tuple:
    """Функция функция обрезки кривой по заданным параметрам.
    Функция обрезает до заданных границ, точно в них попадая.
            :argument x, y: массивы кривых
            :argument x_min: левая граница обрезки
            :argument x_max: правая граница обрезки
    """
    i_min_1, = np.where(x >= x_min)
    i_min_1 = i_min_1[0]
    i_max_1, = np.where(x <= x_max)
    i_max_1 = i_max_1[-1]

    x_cut = x[i_min_1:i_max_1]
    y_cut = y[i_min_1:i_max_1]

    if x_cut[1] > x_min:
        x_cut = np.hstack(([x_min], x_cut))
        y_cut = np.hstack(([np.interp(x_min, x, y)], y_cut))

    if x_cut[-1] < x_max:
        x_cut = np.hstack((x_cut, [x_max]))
        y_cut = np.hstack((y_cut, [np.interp(x_max, x, y)]))

    return (x_cut, y_cut)

def compare(x1: np.array, y1: np.array, x2: np.array, y2: np.array, order: int = 50) -> tuple:
    """Функция функция сравнения двух кривых.
            :argument x1, y1: первая кривая
            :argument x2, y2: вторая кривая
            :argument order: количество точек аппроксимирующего сплайна
            :return разностная кривая
        """
    # Определяем границы массива, в которых определены обе кривые
    x_min = max(np.min(x1), np.min(x2))
    x_max = min(np.max(x1), np.max(x2))

    # Обрезаем первую и вторую кривую до определенных границ по массиву x
    x1, y1 = cut(x1, y1, x_min, x_max)
    x2, y2 = cut(x2, y2, x_min, x_max)

    # Массив x для кривой разности
    x = np.linspace(x_min, x_max, order)
    y_min = np.zeros(order)
    y_max = np.zeros(order)

    # Заполнение массивов y_min и y_max разности кривых
    for i in range(order):
        y1_interp = np.interp(x[i], x1, y1)
        y2_interp = np.interp(x[i], x2, y2)
        y_min[i] = min([y1_interp, y2_interp])
        y_max[i] = max([y1_interp, y2_interp])

    return (x, y_min, y_max)

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    x1 = np.linspace(0, 10, 100)
    y1 = 3 * x1
    x2 = np.linspace(1, 9, 30)
    y2 = x2 ** 2

    x, y_min, y_max = function_compare(x1, y1, x2, y2)

    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.fill_between(x, y_min, y_max, alpha=0.2, color="tomato")
    plt.show()