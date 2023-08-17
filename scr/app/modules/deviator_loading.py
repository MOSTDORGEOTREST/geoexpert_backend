# coding: utf-8

# In[34]:
import copy
import os

from intersect import intersection
from scipy.special import comb
from scipy.optimize import fsolve
import math
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QFrame, QSlider, QLabel
from PyQt5 import QtCore
from scipy import interpolate

import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.pyplot import gca

import numpy as np
import math
from scipy.interpolate import make_interp_spline
from scipy.interpolate import splev, splrep
from scipy.signal import argrelextrema


def create_deviation_curve(x, amplitude, val = (1, 1), points = False, borders = False, one_side = False, low_first_district = False):
    """Возвращает рандомную кривую с размерностью, как x.
    Входные параметры:  :param x: входной массив
                        :param amplitude: размах
                        :param val: значение в первой и последней точке
                        :param points: количество точек изгиба
                        :param borders: условие производной на границах. чтобы было 0 подаем 'zero_diff'
                        :param one_side: делает кривую со значениями больше 0. Подается True, False
                        :param low_first_district: задает начальный участок с меньшими значениями,
                         чтобы не было видно скачка производной. Подается как число начальных участков"""

    def random_value_in_array(x):
        """Возвращает рандомное значение в пределах массива"""
        return np.random.uniform(min(x), max(x))

    def split_amplitude(amplitude, points_of_deviations_count, val, low_first_district):
        """Делает линейную зависимость амплитуды от точки"""
        x = np.linspace(0, points_of_deviations_count - 1, points_of_deviations_count)

        if low_first_district:
            return np.hstack((np.array([amplitude / 10 for _ in range(low_first_district)]), np.array(
                [((i / x[-1]) * (val[1] * amplitude - val[0] * amplitude)) + val[0] * amplitude for i in
                 x[low_first_district:]])))
        else:
            return np.array([((i / x[-1]) * (val[1] * amplitude - val[0] * amplitude)) + val[0] * amplitude for i in x])

    def create_amplitude_array(amplitude, x, val):
        """Делает массив с линейной зависимостью амплитуды от точки"""
        return np.linspace(amplitude*val[0], amplitude*val[1], len(x))

    # Определим количество точек условного перегиба
    if points:
        points_of_deviations_count = int(points)
    else:
        points_of_deviations_count = int(np.random.uniform(5, 10))

    if low_first_district:
        if low_first_district > points - 1:
            low_first_district = points - 1

    def create_deviations_array(amplitude, points_of_deviations_count, val, low_first_district):
        """Делает массив с линейной зависимостью амплитуды от точки"""

        # Определим начение y в каждой точке перегиба
        if one_side:
            y_points_of_deviations_array = np.hstack((0, np.array([np.random.uniform(amp/3, amp) for amp in split_amplitude(amplitude, points_of_deviations_count, val, low_first_district)]), 0))
        else:
            y_points_of_deviations_array = np.hstack((0, np.array(
                [np.random.uniform(-amp, amp) for amp in split_amplitude(amplitude, points_of_deviations_count, val, low_first_district)]), 0))

        # Определим начение x в каждой точке перегиба
        x_points_of_deviations_array = np.hstack((x[0],
                                                  np.array([random_value_in_array(i) for i in np.hsplit(x[:int(points_of_deviations_count*(len(x)//points_of_deviations_count))], points_of_deviations_count)]),
                                                           x[-1]))
        return x_points_of_deviations_array, y_points_of_deviations_array

    x_points_of_deviations_array, y_points_of_deviations_array = create_deviations_array(amplitude, points_of_deviations_count, val, low_first_district)

    # Создадим сплайн
    if borders == "zero_diff":
        iterpolate_curve = make_interp_spline(x_points_of_deviations_array, y_points_of_deviations_array, k=3,
                                              bc_type="clamped")
        deviation_curve = iterpolate_curve(x)

        if one_side:
            while min(deviation_curve)<0:
                x_points_of_deviations_array, y_points_of_deviations_array = \
                    create_deviations_array(amplitude, points_of_deviations_count, val, low_first_district)
                iterpolate_curve = make_interp_spline(x_points_of_deviations_array, y_points_of_deviations_array, k=3,
                                                      bc_type="clamped")
                deviation_curve = iterpolate_curve(x)

    else:
        iterpolate_curve = splrep(x_points_of_deviations_array, y_points_of_deviations_array, k=3)
        deviation_curve = np.array(splev(x, iterpolate_curve, der=0))

        if one_side:
            while min(deviation_curve)<0:
                x_points_of_deviations_array, y_points_of_deviations_array = \
                    create_deviations_array(amplitude, points_of_deviations_count, val, low_first_district)
                iterpolate_curve = splrep(x_points_of_deviations_array, y_points_of_deviations_array, k=3)
                deviation_curve = np.array(splev(x, iterpolate_curve, der=0))

    # Нормируем сплайнн
    if amplitude != 0:
        amplitude_array = create_amplitude_array(amplitude, x, val)
        imax = np.argmax((np.abs(deviation_curve) + min(np.abs(deviation_curve)/1000))/amplitude_array)
        if deviation_curve[imax] != 0 and amplitude_array[imax]!=0:
            deviation_curve /= deviation_curve[imax]/amplitude_array[imax]

    #xvals, yvals = bezier_curve(x, deviation_curve, nTimes=50)

    return deviation_curve

def define_E50_qf(strain, deviator):
    """Определение параметров qf и E50"""
    qf = np.max(deviator)

    imax = 0
    for i in range(len(deviator)):
        if deviator[i] > qf / 2:
            imax = i
            break

    if imax == 0:
        imax = len(deviator)

    imin = imax - 1

    E50 = (qf / 2) / (
        np.interp(qf / 2, np.array([deviator[imin], deviator[imax]]), np.array([strain[imin], strain[imax]])))

    return E50, qf


def discrete_array(array, n_step):
    """Функция делает массив дискретным по заданнаму шагу датчика
    Входные параметры: array - массив данных
    n_step - значение шага"""
    current_val = (array[0]//n_step)*n_step # значение массива с учетом шага в заданной точке
    for i in range(1, len(array)): # перебираем весь массив
        count_step = (array[i]-current_val)//n_step
        array[i] = current_val + count_step*n_step
        current_val = array[i]
    return array


def deviator_loading_deviation1(strain, deviator, xc, amplitude):
    # Добавим девиации после 0.6qf для кривой без пика
    qf = max(deviator)

    # Добавим девиации после 0.6qf для кривой без пика
    index_015, = np.where(strain >= 0.15)
    qf = np.max(deviator[:index_015[0]])
    amplitude_1, amplitude_2, amplitude_3 = amplitude

    points_1 = np.random.uniform(7, 12)
    points_2 = np.random.uniform(20, 30)
    points_3 = np.random.uniform(50, 60)

    devition_1 = amplitude_1 * qf
    devition_2 = amplitude_2 * qf
    devition_3 = amplitude_3 * qf

    i_60, = np.where(deviator >= 0.51 * qf)
    i_90, = np.where(deviator >= 0.98 * qf)
    i_end, = np.where(strain >= 0.15)
    i_xc, = np.where(strain >= xc)
    if xc >= 0.14:  # без пика
        try:
            curve = create_deviation_curve(strain[i_60[0]:i_xc[0]], devition_1,
                                           points=points_1, borders="zero_diff",
                                           low_first_district=1, one_side=True) + \
                    create_deviation_curve(strain[i_60[0]:i_xc[0]], devition_2,
                                           points=points_2, borders="zero_diff",
                                           low_first_district=1, one_side=True) + \
                    сreate_deviation_curve(strain[i_60[0]:i_xc[0]], devition_3,
                                           points=points_3, borders="zero_diff",
                                           low_first_district=1, one_side=True)
            deviation_array = -np.hstack((np.zeros(i_60[0]),
                                          curve,
                                          np.zeros(len(strain) - i_xc[0])))
        except IndexError:
            deviation_array = np.zeros(len(strain))

    else:

        try:
            i_xc1, = np.where(deviator[i_xc[0]:] <= qf - devition_2)
            i_xc_m, = np.where(deviator >= qf - devition_1 * 2)
            points_1 = round((xc) * 30)
            if points_1 < 3:
                points_1 = 3
            points_2 = round(points_1 * np.random.uniform(1.5, 2))
            points_3 = round(points_1 * np.random.uniform(2.5, 3))

            curve_1 = create_deviation_curve(strain[i_60[0]:i_xc_m[0]], devition_1 / 3,
                                             points=points_1, val=(1, 0.1), borders="zero_diff",
                                             low_first_district=1) + \
                      create_deviation_curve(strain[i_60[0]:i_xc_m[0]], devition_2 / 3,
                                             points=points_2, borders="zero_diff",
                                             low_first_district=1) + \
                      create_deviation_curve(strain[i_60[0]:i_xc_m[0]], devition_3 / 3,
                                             points=points_3, borders="zero_diff",
                                             low_first_district=1)

            points_2 = round((0.15 - xc) * 100)
            if points_2 < 3:
                points_2 = 3
            points_2 = round(points_1 * np.random.uniform(2, 3))
            points_3 = round(points_1 * np.random.uniform(5, 6))

            # devition_2 = ((deviator[i_xc[0]] - deviator[i_end[0]]) / 14) * (points_2 / 10)

            curve_2 = create_deviation_curve(strain[i_xc[0] + i_xc1[0]:i_end[0]],
                                             devition_1, val=(0.1, 1),
                                             points=points_1, borders="zero_diff",
                                             low_first_district=2) + create_deviation_curve(
                strain[i_xc[0] + i_xc1[0]:i_end[0]],
                devition_2, val=(0.1, 1),
                points=points_2, borders="zero_diff",
                low_first_district=2) + create_deviation_curve(
                strain[i_xc[0] + i_xc1[0]:i_end[0]],
                devition_3, val=(0.1, 1),
                points=points_3, borders="zero_diff",
                low_first_district=2)
            deviation_array = -np.hstack((np.zeros(i_60[0]),
                                          curve_1, np.zeros(i_xc[0] - i_xc_m[0]),
                                          np.zeros(i_xc1[0]),
                                          curve_2,
                                          np.zeros(len(strain) - i_end[0])))
        except (ValueError, IndexError):
            print("Ошибка девиаций девиатора")
            deviation_array = -np.hstack((np.zeros(i_60[0]),
                                          create_deviation_curve(strain[i_60[0]:i_90[0]], devition_1,
                                                                 points=np.random.uniform(3, 6), borders="zero_diff",
                                                                 low_first_district=1),
                                          create_deviation_curve(strain[i_90[0]:i_end[0]], devition_2, val=(1, 0.1),
                                                                 points=np.random.uniform(10, 15), borders="zero_diff",
                                                                 low_first_district=3,
                                                                 one_side=True),
                                          np.zeros(len(strain) - i_end[0])))

    return deviation_array


def deviator_loading_deviation1_old(strain, deviator, xc, amplitude):
    # Добавим девиации после 0.6qf для кривой без пика
    qf = max(deviator)

    devition_1 = amplitude * qf
    devition_2 = amplitude * qf * 0.6

    i_60, = np.where(deviator >= 0.51 * qf)
    i_90, = np.where(deviator >= 0.98 * qf)
    i_end, = np.where(strain >= 0.15)
    i_xc, = np.where(strain >= xc)
    if xc >= 0.14:  # без пика
        try:
            curve = create_deviation_curve(strain[i_60[0]:i_xc[0]], devition_1 / 2,
                                           points=np.random.uniform(6, 15), borders="zero_diff",
                                           low_first_district=1, one_side=True) + \
                    create_deviation_curve(strain[i_60[0]:i_xc[0]], devition_1,
                                           points=np.random.uniform(20, 30), borders="zero_diff",
                                           low_first_district=1, one_side=True)
            deviation_array = -np.hstack((np.zeros(i_60[0]),
                                          curve,
                                          np.zeros(len(strain) - i_xc[0])))
        except IndexError:
            deviation_array = np.zeros(len(strain))

    else:

        try:
            i_xc1, = np.where(deviator[i_xc[0]:] <= qf - devition_2)
            i_xc_m, = np.where(deviator >= qf - devition_1 * 2)
            points_1 = round((xc) * 100)
            if points_1 < 3:
                points_1 = 3

            curve_1 = create_deviation_curve(strain[i_60[0]:i_xc_m[0]], devition_1 * 1.5,
                                             points=np.random.uniform(3, 4), val=(1, 0.1), borders="zero_diff",
                                             low_first_district=1) + create_deviation_curve(
                strain[i_60[0]:i_xc_m[0]], devition_1 / 2,
                points=np.random.uniform(points_1, points_1 * 3), borders="zero_diff",
                low_first_district=1)

            points_2 = round((0.15 - xc) * 100)
            if points_2 < 3:
                points_2 = 3

            devition_2 = ((deviator[i_xc[0]] - deviator[i_end[0]]) / 14) * (points_2 / 10)

            curve_2 = create_deviation_curve(strain[i_xc[0] + i_xc1[0]:i_end[0]],
                                             devition_2, val=(0.1, 1),
                                             points=np.random.uniform(points_2, int(points_2 * 3)), borders="zero_diff",
                                             low_first_district=2) + create_deviation_curve(
                strain[i_xc[0] + i_xc1[0]:i_end[0]],
                devition_2 / 3, val=(0.1, 1),
                points=np.random.uniform(points_2 * 3, int(points_2 * 5)), borders="zero_diff",
                low_first_district=2)
            deviation_array = -np.hstack((np.zeros(i_60[0]),
                                          curve_1, np.zeros(i_xc[0] - i_xc_m[0]),
                                          np.zeros(i_xc1[0]),
                                          curve_2,
                                          np.zeros(len(strain) - i_end[0])))
        except (ValueError, IndexError):
            print("Ошибка девиаций девиатора")
            deviation_array = -np.hstack((np.zeros(i_60[0]),
                                          create_deviation_curve(strain[i_60[0]:i_90[0]], devition_1,
                                                                 points=np.random.uniform(3, 6), borders="zero_diff",
                                                                 low_first_district=1),
                                          create_deviation_curve(strain[i_90[0]:i_end[0]], devition_2, val=(1, 0.1),
                                                                 points=np.random.uniform(10, 15), borders="zero_diff",
                                                                 low_first_district=3,
                                                                 one_side=True),
                                          np.zeros(len(strain) - i_end[0])))

    return deviation_array


def deviator_loading_deviation(strain, deviator, xc, amplitude):
    # Добавим девиации после 0.6qf для кривой без пика
    index_015, = np.where(strain >= 0.15)
    qf = np.max(deviator[:index_015[0]])
    amplitude_1, amplitude_2, amplitude_3 = amplitude

    devition_1 = amplitude_1 * qf
    devition_2 = amplitude_2 * qf
    devition_3 = amplitude_3 * qf

    try:
        index_015, = np.where(strain >= 0.17)
        index_015 = index_015[0]
    except TypeError:
        index_015 = -1

    points_1 = np.random.uniform(7, 12)
    points_2 = np.random.uniform(20, 30)
    points_3 = np.random.uniform(50, 60)

    if xc < 0.14:
        i_max = np.argmax(deviator)
        i, = np.where(deviator[i_max:] <= (1 - 0.08) * np.max(deviator))
        if len(i) != 0:
            index_008 = i[0] + i_max
            k = 1 / (strain[index_008] / strain[index_015])
        else:
            k = 1
        points_1 = int(points_1 * k * 0.7)
        points_2 = int(points_2 * k * 0.7)
        points_3 = int(points_3 * k * 0.7)

    try:
        strain_for_deviations = strain[:index_015]
        curve_1 = create_deviation_curve(strain_for_deviations, devition_1,
                                         points=points_1, borders="zero_diff",
                                         low_first_district=1, one_side=True)
        curve_2 = create_deviation_curve(strain_for_deviations, devition_2,
                                         points=points_2, borders="zero_diff",
                                         low_first_district=1, one_side=True)
        curve_3 = create_deviation_curve(strain_for_deviations, devition_3,
                                         points=points_3, borders="zero_diff",
                                         low_first_district=1, one_side=True)
        deviation_array = -(curve_1 + curve_2 + curve_3)
        deviation_array = np.hstack((deviation_array, np.zeros(len(strain[index_015:]))))
    except IndexError:
        deviation_array = np.zeros(len(strain))


    except (ValueError, IndexError):
        print("Ошибка девиаций девиатора")
        pass

    return deviation_array


def deviation_volume_strain1(x, x_given, xc, len_x_dilatacy, deviation=0.0015):
    index_x_given, = np.where(x >= x_given)
    n = xc / 0.15
    index_x_start_dilatacy, = np.where(x >= (xc - len_x_dilatacy * 2))
    index_x_end_dilatacy, = np.where(x >= (xc + len_x_dilatacy * 2))

    if xc >= 0.14:
        deviation_vs = np.hstack((np.zeros(index_x_given[0]),
                                  create_deviation_curve(x[index_x_given[0]:], deviation * 2 * n,
                                                         points=np.random.uniform(5, 10),
                                                         val=(0.3, 1), borders='zero diff') + create_deviation_curve(
                                      x[index_x_given[0]:],
                                      deviation * 0.7 * n, points=np.random.uniform(15, 30),
                                      val=(0.3, 1), borders='zero diff')))
        return deviation_vs

    if xc <= 0.03:
        deviation_vs = np.hstack((np.zeros(index_x_given[0]),
                                  create_deviation_curve(x[index_x_given[0]:], deviation * 2 * n,
                                                         points=np.random.uniform(5, 10),
                                                         val=(0.3, 1), borders='zero diff') + create_deviation_curve(
                                      x[index_x_given[0]:],
                                      deviation * 0.7 * n, points=np.random.uniform(15, 30),
                                      val=(0.3, 1), borders='zero diff')))
        return deviation_vs

    try:
        deviation_vs = np.hstack((np.zeros(index_x_given[0]),
                                  create_deviation_curve(x[index_x_given[0]:index_x_start_dilatacy[0]],
                                                         deviation * 2 * n, points=np.random.uniform(5, 10),
                                                         val=(0.3, 1), borders='zero diff') + create_deviation_curve(
                                      x[index_x_given[0]:index_x_start_dilatacy[0]],
                                      deviation * 0.7 * n, points=np.random.uniform(15, 30),
                                      val=(0.3, 1), borders='zero diff'),

                                  np.zeros(len(x[index_x_start_dilatacy[0]:index_x_end_dilatacy[0] + 1])),
                                  create_deviation_curve(x[index_x_end_dilatacy[0] + 1:], deviation, val=(0.3, 1),
                                                         borders='zero diff')))
        return deviation_vs

    except (ValueError, IndexError):
        deviation_vs = np.hstack((np.zeros(len(x[:index_x_given[0]])),
                                  create_deviation_curve(x[index_x_given[0]:index_x_start_dilatacy[0]], deviation / 8,
                                                         val=(1, 0.3), borders='zero diff'),
                                  np.zeros(len(x[index_x_start_dilatacy[0]:index_x_end_dilatacy[0] + 1])),
                                  create_deviation_curve(x[index_x_end_dilatacy[0] + 1:], deviation, val=(0.3, 1),
                                                         borders='zero diff')))
        return deviation_vs


def deviation_volume_strain(x, x_given, xc, len_x_dilatacy, deviation=0.0015):
    index_x_given, = np.where(x >= x_given)
    n = 1
    index_x_start_dilatacy, = np.where(x >= (xc - len_x_dilatacy * 2))
    index_x_end_dilatacy, = np.where(x >= (xc + len_x_dilatacy * 2))
    if xc <= 0.14:
        def deviation_array(x, i_1, i_2, deviation_val, count_1=20, count_2=50):
            points_count = (i_2 - i_1)
            points_1 = int(points_count / count_1)
            points_2 = int(points_count / count_2)

            if (points_1 >= 3) and (points_2 >= 3):
                array = deviation_val * create_deviation_curve(x[i_1: i_2], 1, points=points_1, val=(0.3, 1),
                                                               borders='zero diff') + \
                        deviation * 0.3 * create_deviation_curve(x[i_1: i_2], 1, points=points_2, val=(0.3, 1),
                                                                 borders='zero diff')
            elif (points_1 >= 3) and (points_2 < 3):
                array = deviation_val * create_deviation_curve(x[i_1: i_2], 1, points=points_1, val=(0.3, 1),
                                                               borders='zero diff')
            else:
                array = np.zeros(i_2 - i_1)

            return array

        try:
            starn_puasson = np.zeros(index_x_given[0])
            puasson_start_dilatacy = deviation_array(x, index_x_given[0], index_x_start_dilatacy[0], deviation / 2)
            start_dilatacy_end_dilatacy = deviation_array(x, index_x_start_dilatacy[0], index_x_end_dilatacy[0],
                                                          deviation / 10)
            end_dilatacy_end = deviation_array(x, index_x_end_dilatacy[0], len(x), deviation)

            deviation_vs = np.hstack(
                (starn_puasson, puasson_start_dilatacy, start_dilatacy_end_dilatacy, end_dilatacy_end))
        except IndexError:
            deviation_vs = np.hstack((np.zeros(index_x_given[0]),
                                      deviation * 2 * n * create_deviation_curve(x[index_x_given[0]:], 1,
                                                                                 points=np.random.uniform(5, 10),
                                                                                 val=(0.3, 1), borders='zero diff') +
                                      deviation * 0.7 * n * create_deviation_curve(
                                          x[index_x_given[0]:], 1, points=np.random.uniform(15, 30),
                                          val=(0.3, 1), borders='zero diff')))

        return deviation_vs
    else:
        deviation_vs = np.hstack((np.zeros(index_x_given[0]),
                                  deviation * 2 * n * create_deviation_curve(x[index_x_given[0]:], 1,
                                                                             points=np.random.uniform(5, 10),
                                                                             val=(0.3, 1), borders='zero diff') +
                                  deviation * 0.7 * n * create_deviation_curve(
                                      x[index_x_given[0]:], 1, points=np.random.uniform(15, 30),
                                      val=(0.3, 1), borders='zero diff')))
        return deviation_vs


# Девиаторное нагружение
def params_gip_exp_tg(x, e50, qf, x50, xc, qocr):
    '''возвращает коэффициенты гиперболы, экспоненты и тангенса'''
    kp = np.linspace(0, 1, len(x))  # kp - коэффициент влияния на k, учитывающий переуплотнение qocr

    for i in range(len(x)):
        kp[i] = 1.

    def equations_e(p_e):
        a1_e, k1_e = p_e  # коэффициенты экспоненты
        return -a1_e * (np.exp(-k1_e * x50) - 1) - qf / 2, -a1_e * (np.exp(-k1_e * xc) - 1) - qf

    # начальные приближения заданные по участкам
    if e50 > 40000:
        a1_e, k1_e = fsolve(equations_e, (1, 1))
        nach_pr_a1_e = 1
        error = equations_e([a1_e, k1_e])
        while abs(error[0]) >= 10 or abs(error[1]) >= 10:
            nach_pr_a1_e += 1
            a1_e, k1_e = fsolve(equations_e, (nach_pr_a1_e, 1))
            error = equations_e([a1_e, k1_e])

    else:
        nach_pr_a1_e, nach_pr_k1_e = 600, 1
        result = fsolve(equations_e, (nach_pr_a1_e, nach_pr_k1_e), full_output=1)
        a1_e, k1_e = result[0]

        bad_progress_count = 0  # если в result[2] находится 4 или 5, то вылезает предупреждение с прохой сходимостью
        # мы будем делать 50 итераций чтобы попытаться избавиться от этого предупреждения

        nach_pr_a1_e = 1
        error = equations_e([a1_e, k1_e])
        count = 0
        while (a1_e <= 0) or (k1_e <= 0) or (a1_e == nach_pr_a1_e or k1_e == nach_pr_k1_e) or (
                a1_e * k1_e < 1000) or (
                (abs(error[0]) >= 550 or abs(error[1]) >= 550) and
                count < 50) or (
                result[2] in (4, 5) and bad_progress_count < 50 and qf < 251):  # если начальное приближение a1_e
            # или k1_e равно 1 или произведение коээфициентов (приблизительно равно e50)
            #  меньше 1000, то ищутся другие начальные приближения
            nach_pr_a1_e += 1
            result = fsolve(equations_e, (nach_pr_a1_e, nach_pr_k1_e), full_output=1)
            a1_e, k1_e = result[0]

            if result[2] in (4, 5):
                bad_progress_count += 1

            error = equations_e([a1_e, k1_e])
            if (abs(error[0]) >= 550 or abs(error[1]) >= 550):
                count += 1

    # коэффициенты гиперболы
    k1_g = -2 / xc + 1 / x50
    a1_g = qf * (k1_g * x50 + 1.) / (2. * x50)

    def equations_t(p_t):
        a1_t, k1_t = p_t  # коэффициенты тангенса
        return ((a1_t * ((np.arctan(k1_t * x50)) / (0.5 * np.pi)) - qf / 2),
                (a1_t * ((np.arctan(k1_t * xc)) / (0.5 * np.pi)) - qf))

    if e50 > 50000:
        a1_t, k1_t = fsolve(equations_t, (1, 1))
    else:
        a1_t, k1_t = fsolve(equations_t, (600, 1))

    xocr = 0.  # абцисса точки переуплотнения
    if (qocr != 0.) & (qocr <= qf / 2.):  # если qocr находится до qf/2, то xocr рассчитывается
        # из функции гиперболы
        xocr = qocr / (a1_g - qocr * k1_g)
        for i in range(len(x)):
            kp[i] = form_kp(x[i], qf, k, xocr, xc, qocr, x50)

    elif (qocr != 0.) & (qocr > qf / 2.) & (e50 <= 70000):  # если qocr находится после qf/2 и e50 находится до 70000,
        # то xocr рассчитывается из функции экспоненты
        def equations_xocr(xocr):
            return (-a1_e * (np.exp(-k1_e * xocr) - 1) - qocr)

        xocr = fsolve(equations_xocr, 0)
        for i in range(len(x)):
            kp[i] = form_kp(x[i], qf, k, xocr, xc, qocr, x50)

    elif (qocr != 0.) & (qocr > qf / 2.) & (e50 > 70000):  # если qocr находится после qf/2 и e50 находится после 70000,
        # то xocr рассчитывается из функции тангенса, так как переуплотнение плавнее
        def equations2(xocr):
            return (a1_t * ((np.arctan(k1_t * xocr)) / (0.5 * np.pi)) - qocr)

        xocr = fsolve(equations2, 0)
        for i in range(len(x)):
            kp[i] = form_kp(x[i], qf, k, xocr, xc, qocr, x50)

    elif qocr > (0.8 * qf):  # ограничение на qocr (если ограничение не выполняется, то
        # строится сумма функций экспоненты и кусочной функции синуса и параболы для e50<=70000
        # и тангенса и кусочной функции синуса и параболы для e50>70000
        for i in range(len(x)):
            kp[i] = 0.

    return a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp, xocr


def hevisaid(x, sdvig, delta_x):
    ''' возвращет функцию Хевисайда, которая задает коэффициент влияния kp'''
    return 1. / (1. + np.exp(-2 * 10 / delta_x * (x - sdvig)))


def gip_and_exp_or_tg(x, e50, x50, qf, a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp, k, qocr,
                      xocr):
    '''сумма функций гиперболы и экспоненты с учетом коэффициентов влияния'''

    ret = ((kp * k) * (a1_g * x / (1 + k1_g * x)) + (
            (1. - kp * k) * (-a1_e * (np.exp(-k1_e * x) - 1))))  # сумма гиперболы и экспоненты

    if (qocr > qf / 2.) & (qocr != 0.) & (x > x50) & (e50 <= 70000):
        ret = ((kp * k) * (a1_g * x / (1 + k1_g * x)) + (
                (1. - kp * k) * (-a1_e * (np.exp(-k1_e * x) - 1))))  # сумма гиперболы и экспоненты (x>x50)

    elif (qocr > qf / 2.) & (qocr != 0.) & (x > x50) & (e50 > 70000):
        ret = ((kp * k) * (a1_g * x / (1 + k1_g * x)) + (
                (1. - kp * k) * (
                a1_t * ((np.arctan(k1_t * x)) / (0.5 * np.pi)))))  # сумма гиперболы и тангенса (x>x50)
    elif (qocr > 0.8 * qf) & (qocr != 0.):
        ret = ((kp * k) * (a1_g * x / (1 + k1_g * x)) + (
                (1. - kp * k) * (-a1_e * (np.exp(
            -k1_e * x) - 1))))  # в случае невыполнения ограничения на qocr, строится только экспонента (kp=0)

    if (qocr <= qf / 2.) & (qocr != 0.) & (
            x <= x50):
        if x <= xocr:  # сумма функций с учетом коэффициентов влияния до xocr (x<x50)
            ret = ((1. - kp * k) * (a1_g * x / (1 + k1_g * x)) + (
                    (kp * k) * (-a1_e * (np.exp(-k1_e * x) - 1))))  # сумма гиперболы и экспоненты
        else:  # сумма функций с учетом коэффициентов влияния после xocr (x<x50)
            ret = ((1. - kp * (1 - k)) * (a1_g * x / (1 + k1_g * x)) + (
                    (kp * (1 - k)) * (-a1_e * (np.exp(-k1_e * x) - 1))))  # сумма гиперболы и экспоненты

    return ret


def cos_par(x, e50, qf, x50, xc, hlow):
    '''возвращает функцию косинуса
     и параболы для участка x50 qf'''

    sm = (xc - x50) / 2  # смещение
    # коэффицент учитывающий влияние на высоту функции при различных значениях e50
    if e50 < 5340:
        vl = 0
    elif (e50 <= 40000) and (e50 >= 5340):
        kvl = 1 / 34660
        bvl = -5340 * kvl
        vl = kvl * e50 + bvl  # 1. / 40000. * e50 - 1. / 8
    elif e50 > 40000:
        vl = 1.

    h = 0.035 * qf * vl - hlow  # высота функции
    if h < 0:
        h = 0

    k = h / (-xc + x50 + sm) ** 2

    # фиромирование функции
    if x < x50:
        cos_par = 0
    elif (x >= x50) and (x <= x50 + sm):
        cos_par = h * (1 / 2) * (np.cos((1. / sm) * np.pi * (x - x50) - np.pi) + 1)  # косинус
    elif (x > x50 + sm) and (x < xc):
        cos_par = -k * (x - x50 - sm) ** 2 + h  # парабола
    elif x >= xc:
        cos_par = 0

    return cos_par


def gaus(x, qf, xc, x2, qf2):
    '''функция Гаусса для участка x>xc'''
    a_gaus = qf - qf2  # высота функции Гаусса
    k_gaus = (-1) * np.log(0.1 / a_gaus) / ((x2 - xc) ** 2)  # резкость функции Гаусаа
    # (считается из условия равенства заданной точности в точке х50
    return a_gaus * (np.exp(-k_gaus * ((x - xc) ** 2))) + qf2


def parab(x, qf, xc, x2, qf2):
    '''функция Гаусса для участка x>xc'''
    k_par = -((qf2 - qf) / (x2 - xc) ** 2)
    return -k_par * ((x - xc) ** 2) + qf


def smoothness_condition(qf, x50):
    '''возвращает предельное значение xc при котором возможно
    построение заданной функции'''
    k_lim = qf / (2 * x50)
    x_lim = (qf / k_lim)

    x_lim += 0.6 / 100

    return x_lim


def form_kp(x: float, qf, k, xocr, xc, qocr, x50):
    '''вовзращает коэффициент влияния kp'''
    kp = 1.
    if qocr > qf / 2.:  # если qsr на участке от qf / 2

        if qocr > (0.8 * qf):  # ограничение на qocr если ограничение не выполняется, то
            # строится экспоненты
            kp = 0.

        else:
            if x <= xocr:  # принудительное зануление отрицательной части функции Хевисайда
                kp = 0.
            elif (x > xocr) & (x <= xc):
                delta = ((abs(xocr - x50)) / (k + 0.000001)) * 10  # ширина Хевисайда
                kp = 2 * hevisaid(x, xocr, delta) - 1.

    elif (qocr < qf / 2.) & (qocr != 0.):  # если qsr на участке до qf / 2.
        if x > xocr:

            delta = abs(10 * (x50 - xocr + 0.000001))  # ширина Хевисайда
            kp = 2 * hevisaid(x, xocr, delta) - 1
        else:
            kp = 0.  # принудительное зануление отрицательной части функции Хевисайда

    elif qocr == 0.:  # для нулевого переуплотнения
        kp = 1.

    elif qocr == qf / 2.:  # для qocr == qf/2.
        if x <= xocr:  # до xsr строится функция гиперболы
            kp = 0.
        if x > xocr:
            kp = 1.  # после хsr строится функция суммы гиперболы и экспоненты
            # или тангенса с учетом коэффициентов влияния
    if (xocr == 0.) & (qocr <= (0.8 * qf)):
        kp = 1.

    return kp


def sensor_accuracy(x, y, qf, x50, xc):
    '''возвразщает зашумеленную функцию без шума в характерных точках'''

    x = np.asarray(x)
    y = np.asarray(y)

    max_y = np.max(y)

    sh = np.random.uniform(-0.4, 0.4, len(x))
    index_qf_half, = np.where(y >= max_y / 2)
    index_qf, = np.where(y >= max_y)

    max_x = np.max(x)

    if xc > max_x:  # если хс последня точка в массиве или дальше
        index_qf, = np.where(x >= max_x)

    y_res = y + sh

    # пропускаем нужные точки
    y_res[index_qf_half[0] - 2] = y[index_qf_half[0] - 2]
    y_res[index_qf_half[0] - 1] = y[index_qf_half[0] - 1]
    y_res[index_qf_half[0] - 0] = y[index_qf_half[0] - 0]
    y_res[index_qf_half[0] + 1] = y[index_qf_half[0] + 1]
    y_res[index_qf_half[0] + 2] = y[index_qf_half[0] + 2]

    y_res[index_qf[0] - 2] = y[index_qf[0] - 2]
    y_res[index_qf[0] - 1] = y[index_qf[0] - 1]
    y_res[index_qf[0] - 0] = y[index_qf[0] - 0]

    try:
        y_res[index_qf[0] + 1] = y[index_qf[0] + 1]
    except IndexError:
        pass
    try:
        y_res[index_qf[0] + 2] = y[index_qf[0] + 2]
    except IndexError:
        pass

    # в районе максимума шум меньше первоначального
    indexes, = np.where(y_res > max_y)
    if len(indexes) > 0:
        for i in indexes:
            y_res[i] = y[i] - np.random.uniform(0.05, 0.025)
    # print(indexes, index_qf_half[0], index_qf[0])

    # y_test = copy.deepcopy(y)
    # for i in range(len(y_test)):  # наложение шума кроме промежутков для характерных точек
    #     if (i < index_qf_half[0] - 2) or ((i > index_qf_half[0] + 2) and ([i] < index_qf[0] - 2)) or (
    #             i > index_qf[0] + 2):
    #         if (y_test[i] + sh[i] < np.max(y_test)):
    #             y_test[i] = y_test[i] + sh[i]
    #         else:
    #             y_test[i] = y_test[i] - np.random.uniform(0.05, 0.025)  # в районе максимума шум меньше первоначального
    #
    # for i in range(len(y_test)):
    #     errr = abs(y_test[i] - y_res[i])
    #     if errr > 0.05:
    #         print(i, errr)

    return y_res


def interpolated_intercepts(x, y1, y2):
    """Find the intercepts of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0] * p2[1] - p2[0] * p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x, y

        L1 = line([point1[0], point1[1]], [point2[0], point2[1]])
        L2 = line([point3[0], point3[1]], [point4[0], point4[1]])

        R = intersection(L1, L2)

        return R

    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

    xcs = []
    ycs = []

    for idx in idxs:
        xc, yc = intercept((x[idx], y1[idx]), ((x[idx + 1], y1[idx + 1])), ((x[idx], y2[idx])),
                           ((x[idx + 1], y2[idx + 1])))
        xcs.append(xc)
        ycs.append(yc)
    return np.array(xcs), np.array(ycs)

def bezier_curve(p1_l1, p2_l1, p1_l2, p2_l2, node1, node2, x_grid):
    """
    Требуется модуль: from scipy.optimize import fsolve
    Функция построения кривой Безье на оссновании двух прямых,
    задаваемых точками 'point_line' типа [x,y],
    на узлах node типа [x,y]
    с построением промежуточного узла в точке пересечения поданных прямых.
    Функция возвращает значения y=f(x) на сетке по оси Ox.

    Пример:
    Соединяем две прямые от точки [x_given,y_given] до точки [x[index_x_start[0]], y_start[0]]
    xgi, = np.where(x > x_given) # некая точка после которой нужно переходить к кривой безье
    y_Bezier_line = bezier_curve([0,0],[x_given,y_given], #Первая и Вторая точки первой прямой
                                 [x_given, k * x_given + b], #Первая точка второй прямой (k и b даны)
                                 [x[index_x_start[0]],y_start[0]], #Вторая точка второй прямой
                                 [x_given, y_given], #Первый узел (здесь фактически это 2 точка первой прямой)
                                 [x[index_x_start[0]], y_start[0]], #Второй узел
                                                                # (здесь фактически это 2 точка второй прямой)
                                 x[xgi[0]:index_x_start[0]]
                                 )

    :param p1_l1: Первая точка первой прямой [x,y]
    :param p2_l1: Вторая точка первой прямой [x,y]
    :param p1_l2: Первая точка второй прямой [x,y]
    :param p2_l2: Вторая точка второй прямой [x,y]
    :param node1: Первый узел [x,y]
    :param node2: Второй узел [x,y]
    :param x_grid: Сетка по Ох на которой необходимо посчитать f(x)
    :return: Значения y=f(x) на сетке x_grid
    """

    def bernstein_poly(i, n, t):
        """
         Полином Бернштейна стпени n, i - функция t
        """
        return comb(n, i) * (t ** i) * (1 - t) ** (n - i)

    def bezier_curve_local(nodes, n_times=1000):
        """
        На основании набора узлов возвращает
        кривую Безье определяемую узлами
        Точки задаются в виде:
           [ [1,1],
             [2,3],
              [4,5], ..[Xn, Yn] ]
        nTimes - число точек для вычисления значений
        """

        n_points = len(nodes)
        x_points = np.array([p[0] for p in nodes])
        y_points = np.array([p[1] for p in nodes])

        t = np.linspace(0.0, 1.0, n_times)

        polynomial_array = np.array([bernstein_poly(i, n_points - 1, t) for i in range(0, n_points)])

        x_values_l = np.dot(x_points, polynomial_array)
        y_values_l = np.dot(y_points, polynomial_array)
        return x_values_l, y_values_l

    def intersect(xp1, yp1, xp2, yp2, xp3, yp3, xp4, yp4):
        """
        Функция пересечения двух прямых, заданных точками
        :param xp1: x точки 1 на прямой 1
        :param yp1: y точки 1 на прямой 1
        :param xp2: x точки 2 на прямой 1
        :param yp2: y точки 2 на прямой 1
        :param xp3: x точки 1 на прямой 2
        :param yp3: y точки 1 на прямой 2
        :param xp4: x точки 2 на прямой 2
        :param yp4: y точки 2 на прямой 2
        :return: точка пересечения прямых [x,y]
        """

        def line(xp1, yp1, xp2, yp2):
            k = (yp2 - yp1) / (xp2 - xp1)
            b = yp1 - k * xp1
            return k, b

        kl1, bl1 = line(xp1, yp1, xp2, yp2)
        kl2, bl2 = line(xp3, yp3, xp4, yp4)
        x_p_inter = (bl1 - bl2) / (kl2 - kl1)
        y_p_inter = kl1 * x_p_inter + bl1
        return x_p_inter, y_p_inter

    # Определяем точки пересечения прямых
    xl1, yl1 = intersect(p1_l1[0], p1_l1[1],
                         p2_l1[0], p2_l1[1],
                         p1_l2[0], p1_l2[1],
                         p2_l2[0], p2_l2[1])

    # Строим кривую Безье
    x_values, y_values = bezier_curve_local([node1, [xl1, yl1], node2], n_times=len(x_grid))

    # Адаптация кривой под равномерный шаг по х
    bezier_spline = interpolate.make_interp_spline(x_values, y_values, k=1, bc_type=None)
    y_values = bezier_spline(x_grid)

    return y_values


def cos_ocr(x, y, qf, qocr, xc):
    '''возвращает функцию косинуса
     и параболы для участка x50 qf'''

    index_xocr, = np.where(y > qocr)
    xocr = x[index_xocr[0]]
    proiz_ocr = (y[index_xocr[0] + 1] - y[index_xocr[0]]) / \
                (x[index_xocr[0] + 1] - x[index_xocr[0]])

    count = 0
    while proiz_ocr <= 0 and count < 10:
        proiz_ocr = (y[index_xocr[0] + 1 + count] - y[index_xocr[0]]) / (
                    x[index_xocr[0] + 1 + count] - x[index_xocr[0]])
        count += 1

    # print(f"deviator loading functions : cos_ocr : proiz_ocr = {proiz_ocr}")

    if proiz_ocr < 20000:
        vl_h = 0.3
    elif (proiz_ocr >= 20000) and (proiz_ocr <= 80000):
        kvl = 0.7 / 60000
        bvl = 0.3 - 20000 * kvl
        vl_h = kvl * proiz_ocr + bvl  # 1. / 40000. * e50 - 1. / 8
    elif proiz_ocr > 80000:
        vl_h = 1

    max_y_initial = max(y)

    index_max = np.argmax(y)

    h = 0.2 * qf * vl_h  # высота функции

    if h > 0.8 * qocr:
        h = 0.8 * qocr

    sm = xocr

    k = h / (sm) ** 2

    index_2xocr, = np.where(x >= xc)
    index_xocr, = np.where(x >= xocr)

    cos_par = np.hstack((-k * (x[:index_xocr[0]] - sm) ** 2 + h,
                         h * (1 / 2) * (np.cos(
                             (1. / (xc - sm)) * np.pi * (x[index_xocr[0]:index_2xocr[0]] + (xc - 2 * sm)) - np.pi) + 1),
                         np.zeros(len(x[index_2xocr[0]:]))))

    # proiz_ocr = [(y[i]+cos_par[i] - y[i+1]-cos_par[i + 1])/ (x[i] - x[i + 1]) for i in range(len(x)-1)]
    # plt.plot(x[:-1], proiz_ocr)

    extremums = argrelextrema(y + cos_par, np.greater)

    if len(extremums) < 1 or len(extremums[0]) < 1:
        extremums = [[0]]

    y_ocr = y + cos_par

    count = 0
    while ((max(y + cos_par) > max_y_initial) or (
            (extremums[0][0] < index_max) and y_ocr[extremums[0][0]] > 0.9 * max_y_initial)) and count < 200:

        if max(y + cos_par) > max_y_initial:
            xc = xc - 0.0001
            if xc >= sm:
                index_2xocr, = np.where(x >= xc)
                index_xocr, = np.where(x >= xocr)

                cos_par = np.hstack((-k * (x[:index_xocr[0]] - sm) ** 2 + h,
                                     h * (1 / 2) * (np.cos((1. / (xc - sm)) * np.pi * (
                                             x[index_xocr[0]:index_2xocr[0]] + (xc - 2 * sm)) - np.pi) + 1),
                                     np.zeros(len(x[index_2xocr[0]:]))))
        #
        y_ocr = y + cos_par
        delta = 0.01 * h

        if (extremums[0][0] < index_max) and y_ocr[extremums[0][0]] > 0.95 * max_y_initial:
            h = h - delta
            k = h / (sm) ** 2
            index_2xocr, = np.where(x >= xc)
            index_xocr, = np.where(x >= xocr)

            cos_par = np.hstack((-k * (x[:index_xocr[0]] - sm) ** 2 + h,
                                 h * (1 / 2) * (np.cos((1. / (xc - sm)) * np.pi * (
                                         x[index_xocr[0]:index_2xocr[0]] + (xc - 2 * sm)) - np.pi) + 1),
                                 np.zeros(len(x[index_2xocr[0]:]))))
            extremums = argrelextrema(y + cos_par, np.greater)
        #
        y_ocr = y + cos_par
        count = count + 1
        # print(f"cos_ocr : COUNT : {count}")

    return cos_par


def dev_loading(qf, e50, x50, xc, x2, qf2, gaus_or_par, amount_points, hyp_ratio=None):
    qocr = 0  # !!!
    '''кусочная функция: на участкe [0,xc]-сумма функций гиперболы и
    (экспоненты или тангенса) и кусочной функции синуса и парболы
    на участке [xc...]-половина функции Гаусса или параболы'''
    if hyp_ratio == None:
        hyp_ratio = 1.  # k - линейный коэффициент учета влияния функции гиперболы и экспоненты

        if e50 <= 70000:
            hyp_ratio = 0.95 / 68000. * e50 - 1. / 34
        elif e50 < 2000:
            hyp_ratio = 0
        elif e50 > 70000:
            hyp_ratio = 0.95

    if xc < x50:
        xc = x50 * 1.1  # хс не может быть меньше x50

    max_x = xc + 0.6
    x = np.linspace(0, max_x, int((amount_points * max_x / 0.15) / 4))
    y = np.linspace(0, max_x, int((amount_points * max_x / 0.15) / 4))
    a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp, xocr = params_gip_exp_tg(x, e50, qf, x50, xc,
                                                                     qocr)  # считаем  k1, k, xocr на участке до x50, начальное значение kp
    # считаем предельное значение xc
    for i in range(len(x)):
        xcpr = smoothness_condition(qf, x50)

    if (x50 >= xc):  # если x50>xc, xc сдвигается в 0.15, х2,qf2 перестает учитываться,
        # в качестве функции используется сумма гиперболы, экспоненты или тангенса
        # и функции синуса и параболы
        xc = 0.15
        a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp, xocr = params_gip_exp_tg(x, e50, qf, x50, xc, qocr)
        for i in range(len(x)):
            xcpr = smoothness_condition(qf, x50)
        if xc <= xcpr:  # проверка на условие гладкости, если условие не соблюдается,
            # передвинуть xс в предельное значение
            xc = xcpr
            if (xc > 0.11) and (xc < 0.15):
                xc = 0.15
            a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp, xocr = params_gip_exp_tg(x, e50, qf, x50, xc, qocr)
        for i in range(len(x)):
            y[i] = gip_and_exp_or_tg(x[i], e50, x50, qf, a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp[i], hyp_ratio, qocr,
                                     xocr) + cos_par(x[i], e50, qf, x50,
                                                     xc, 0)  # формирование функции девиаторного нагружения
        x2 = xc  # x2,qf2 не выводится
        qf2 = qf  # x2,qf2 не выводится

    else:
        a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp, xocr = params_gip_exp_tg(x, e50, qf, x50, xc, qocr)
        for i in range(len(x)):
            xcpr = smoothness_condition(qf, x50)  # считаем предельно значение xc
        if xc <= xcpr:
            xc = xcpr
            if (xc > 0.11) and (xc < 0.15):
                xc = 0.15
            a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp, xocr = params_gip_exp_tg(x, e50, qf, x50, xc, qocr)
        if (xc > 0.15):
            a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp, xocr = params_gip_exp_tg(x, e50, qf, x50, xc, qocr)
            for i in range(len(x)):
                y[i] = gip_and_exp_or_tg(x[i], e50, x50, qf, a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp[i], hyp_ratio, qocr,
                                         xocr) + cos_par(x[i], e50, qf, x50, xc,
                                                         0)  # формирование функции девиаторного нагружения
            x2 = xc  # x2,qf2 не выводится
            qf2 = qf  # x2,qf2 не выводится

        else:
            if xc >= 0.8 * x2:  # минимально допустимое расстояния между хс и х2
                x2 = 1.2 * xc
            if qf2 >= qf:  # минимально допустимое расстояние мужду qf2 и qf
                qf2 = 0.98 * qf

            a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp, xocr = params_gip_exp_tg(x, e50, qf, x50, xc, qocr)
            gip_and_exp_or_tg_cos_par = np.linspace(0, max_x, int((amount_points * max_x / 0.15) / 4))
            for i in range(len(x)):
                if x[i] < xc:
                    gip_and_exp_or_tg_cos_par[i] = gip_and_exp_or_tg(x[i], e50, x50, qf, a1_g, k1_g, a1_e, k1_e, a1_t,
                                                                     k1_t,
                                                                     kp[i], hyp_ratio, qocr, xocr) + cos_par(x[i], e50,
                                                                                                             qf, x50,
                                                                                                             xc,
                                                                                                             0)
                else:
                    gip_and_exp_or_tg_cos_par[i] = 0.

            maximum = max(gip_and_exp_or_tg_cos_par)

            for i in range(len(x)):

                if x[i] <= xc:
                    if maximum < (qf + 0.002 * qf):
                        y[i] = gip_and_exp_or_tg(x[i], e50, x50, qf, a1_g, k1_g, a1_e, k1_e, a1_t, k1_t,
                                                 kp[i], hyp_ratio, qocr, xocr) + cos_par(x[i], e50, qf, x50, xc, 0)
                    # если максимум суммарной функции на участке от 0 до хс превышает qf, то уменьшаем
                    # высоту функции синуса и параболы на величину разницы в точке xc
                    elif maximum >= abs(qf + 0.002 * qf):
                        y[i] = gip_and_exp_or_tg(x[i], e50, x50, qf, a1_g, k1_g, a1_e, k1_e, a1_t, k1_t,
                                                 kp[i], hyp_ratio, qocr, xocr) + cos_par(x[i], e50, qf, x50, xc,
                                                                                         abs(
                                                                                             maximum - qf + 2 * 0.002 * qf))
                    else:
                        y[i] = gip_and_exp_or_tg(x[i], e50, x50, qf, a1_g, k1_g, a1_e, k1_e, a1_t, k1_t, kp[i], k, qocr,
                                                 xocr)


                elif (x[i] > xc) & (gaus_or_par == 0):
                    y[i] = gaus(x[i], qf, xc, x2, qf2)
                elif (x[i] > xc) & (gaus_or_par == 1):
                    y[i] = parab(x[i], qf, xc, x2, qf2)

    if qocr > (0.8 * qf):  # не выводить точку xocr, qocr
        xocr = xc
        qocr = qf
    xnew = np.linspace(x.min(), x.max(), int(amount_points * max_x / 0.15))  # интерполяция  для сглаживания в пике
    spl = make_interp_spline(x, y, k=5)
    y_smooth = spl(xnew)

    xold = xnew  # масиив х без учета петли (для обьемной деформации)

    return xold, xnew, y_smooth, qf, xc, x2, qf2, e50


def curve(qf, e50, **kwargs):
    try:
        kwargs["xc"]
    except KeyError:
        kwargs["xc"] = 0.15

    try:
        kwargs["x2"]
    except KeyError:
        kwargs["x2"] = 0.15

    try:
        kwargs["qf2"]
    except KeyError:
        kwargs["qf2"] = qf

    try:
        kwargs["qocr"]
    except KeyError:
        kwargs["qocr"] = 0

    try:
        kwargs["gaus_or_par"]
    except KeyError:
        kwargs["gaus_or_par"] = 0

    try:
        kwargs["max_time"]
    except KeyError:
        kwargs["max_time"] = 500


    try:
        kwargs["y_rel_p"]
    except KeyError:
        kwargs["y_rel_p"] = 0.8 * qf

    try:
        kwargs["point2_y"]
    except KeyError:
        kwargs["point2_y"] = 10

    try:
        kwargs["U"]
    except KeyError:
        kwargs["U"] = None

    try:
        kwargs["amplitude"]
    except KeyError:
        kwargs["amplitude"] = (0.001, 0.001, 0.001, True)

    try:
        kwargs["hyp_ratio"]
    except KeyError:
        kwargs["hyp_ratio"] = None

    xc = kwargs.get('xc')
    x2 = kwargs.get('x2')
    qf2 = kwargs.get('qf2')
    qocr = kwargs.get('qocr')
    gaus_or_par = kwargs.get('gaus_or_par')  # 0 - гаус, 1 - парабола
    max_time = kwargs.get('max_time')
    amplitude_1 = kwargs.get('amplitude')[0]
    amplitude_2 = kwargs.get('amplitude')[1]
    amplitude_3 = kwargs.get('amplitude')[2]
    free_deviations = kwargs.get('amplitude')[3]
    '''флаг, отвечает за наложение девиаций на контрольные точки'''
    hyp_ratio = kwargs.get('hyp_ratio')

    if max_time < 50:
        max_time = 50
    if max_time <= 499:
        amount_points = max_time * 20
        amount_points_for_stock = np.random.uniform(1, 3) * 20
    elif max_time > 499 and max_time <= 2999:
        amount_points = max_time * 2
        amount_points_for_stock = np.random.uniform(5, 10) * 2
    else:
        amount_points = max_time / 3
        amount_points_for_stock = np.random.uniform(15, 20) / 3


    qf_old = qf
    if qf < 150:
        k_low_qf = 250 / qf
        qf = qf * k_low_qf
        e50 = e50 * k_low_qf
        qf2 = qf2 * k_low_qf

    if xc > 0.111:
        xc = 0.15

    # ограничение на qf2
    if qf2 >= qf:
        qf2 = qf
    x50 = (qf / 2.) / e50

    x_old, x, y, qf, xc, x2, qf2, e50 = dev_loading(qf, e50, x50, xc, x2, qf2, gaus_or_par, amount_points,
                                                    hyp_ratio=hyp_ratio)  # x_old - без участка разгрузки, возвращается для обьемной деформации


    if qocr > (0.6 * qf):
        qocr = 0.6 * qf

    cos = cos_ocr(x, y, qf, qocr, xc)

    index_xocr, = np.where(y >= qocr)
    xocr = x[index_xocr[0]]

    y_ocr = y + cos

    index_qf2ocr, = np.where(y_ocr >= qf / 2)

    x_qf2ocr = np.interp(qf / 2, [y_ocr[index_qf2ocr[0] - 1], y_ocr[index_qf2ocr[0]]],
                         [x[index_qf2ocr[0] - 1], x[index_qf2ocr[0]]])

    index_x50, = np.where(x >= x50)

    is_OCR = False
    if cos[index_x50[0]] > 0:
        is_OCR = True
        a = np.interp(x50, [x[index_x50[0] - 1], x[index_x50[0]]], [y_ocr[index_x50[0] - 1], y_ocr[index_x50[0]]])
        delta = abs(a - qf / 2)

        e50_ocr = (qf / 2 - delta) / x50
        x50_ocr = (qf / 2) / e50_ocr
        # index_x50_ocr, = np.where(x >= x50_ocr)
        # x50_ocr = x[index_x50_ocr[0]]

        x_old, x, y_ocr, qf, xc, x2, qf2, e50_ocr = dev_loading(qf, e50_ocr, x50_ocr, xc, x2, qf2, gaus_or_par,
                                                                amount_points, hyp_ratio=hyp_ratio)

        y_ocr = y_ocr + cos

        a = np.interp(x50, [x[index_x50[0] - 1], x[index_x50[0]]], [y_ocr[index_x50[0] - 1], y_ocr[index_x50[0]]])
        n = 0

        while abs((a / x50 - (qf / 2) / x50)) > 50 and n < 30:
            a = np.interp(x50, [x[index_x50[0] - 1], x[index_x50[0]]], [y_ocr[index_x50[0] - 1], y_ocr[index_x50[0]]])

            n = n + 1
            delta_ocr = (a - qf / 2)

            delta = delta + delta_ocr

            e50_ocr = (qf / 2 - delta) / x50
            x50_ocr = (qf / 2) / e50_ocr

            x_old, x, y_ocr, qf, xc, x2, qf2, e50_ocr = dev_loading(qf, e50_ocr, x50_ocr, xc, x2, qf2, gaus_or_par,
                                                                    amount_points, hyp_ratio=hyp_ratio)
            #
            y_ocr = y_ocr + cos

        y = copy.deepcopy(y_ocr)


    # МАСШТАБ
    if qf_old < 150:
        y = y / k_low_qf
        qf = qf / k_low_qf
        e50 = e50 / k_low_qf

    # МАСШТАБ ИЗ-ЗА ДЕВИАЦИЙ - ВОЗВРАЩАЕТ Е50 И QF В НУЖНОЕ МЕСТО
    count = 0
    y_no_noise = copy.deepcopy(y)
    x_no_noise = copy.deepcopy(x)
    count_limit = 10
    while count < count_limit:

        if not free_deviations:
            y += deviator_loading_deviation1(x, y, xc, amplitude=(amplitude_1, amplitude_2, amplitude_3))
            break

        y += deviator_loading_deviation(x, y, xc, amplitude=(amplitude_1, amplitude_2, amplitude_3))


        y = sensor_accuracy(x, y, qf, x50, xc)  # шум на кривой без петли
        y = discrete_array(y, 0.5)  # ступеньки на кривой без петли

        #
        if  is_OCR:
            y_ocr = copy.deepcopy(y)
            index_qf2ocr, = np.where(y_ocr >= qf / 2)
            x_qf2ocr = np.interp(qf / 2, [y_ocr[index_qf2ocr[0] - 1], y_ocr[index_qf2ocr[0]]],
                                 [x[index_qf2ocr[0] - 1], x[index_qf2ocr[0]]])
            delta = x50 / x_qf2ocr

            x = x * delta
            index_x50, = np.where(x >= x50)
            y_qf2ocr = np.interp(x50, [x[index_x50[0] - 1], x[index_x50[0]]],
                                 [y_ocr[index_x50[0] - 1], y_ocr[index_x50[0]]])
            k = y_qf2ocr / (qf / 2)
            y_ocr = y_ocr / k
            y = copy.deepcopy(y_ocr)

        y_round = np.round(y, 3)
        index_x2, = np.where(np.round(x, 6) >= 0.15)

        qf2_max = np.max(y_round[:index_x2[0]])

        delta = (qf) / qf2_max
        y = y * delta
        y_round = np.round(y, 3)
        x_round = np.round(x, 6)
        qf_max = np.max(np.round(y_round[:index_x2[0]], 3))

        i_07qf, = np.where(y_round[:index_x2[0]] > qf_max * 0.7)
        imax, = np.where(y_round[:i_07qf[0]] > qf_max / 2)
        imin, = np.where(y_round[:i_07qf[0]] < qf_max / 2)
        imax = imax[0]
        imin = imin[-1]
        x_qf2 = np.interp(qf_max / 2, [y_round[imin], y_round[imax]], [x_round[imin], x_round[imax]])
        delta = x50 / x_qf2
        x = x * delta

        index_x2, = np.where(np.round(x, 6) >= 0.15)

        if len(index_x2) == 0:
            y = copy.deepcopy(y_no_noise)
            x = copy.deepcopy(x_no_noise)
            break

        RES_E50 = define_E50_qf(x[:index_x2[0]], y[:index_x2[0]])

        if round(abs(RES_E50[0] - e50) / 1000, 1) < 0.4:
            break

        count = count + 1
        if count < count_limit:
            y = copy.deepcopy(y_no_noise)
            x = copy.deepcopy(x_no_noise)

    y[0] = 0.


    index_x2, = np.where(x >= 0.15)
    x = x[:index_x2[0]]
    y = y[:index_x2[0]]
    y[0] = 0.  # искусственное зануление первой точки



    if max_time <= 499:
        time = [i / 20 for i in range(len(x))]
    elif max_time > 499 and max_time <= 2999:
        time = [i / 2 for i in range(len(x))]
    else:
        time = [i * 3 for i in range(len(x))]


    return x, y

def define_xc_qf_E(qf, E50):
    """Функция определяет координату пика в зависимости от максимального девиатора и модуля"""
    try:
        k = E50 / qf
    except (ValueError, ZeroDivisionError):
        return 0.15

    # Если все норм, то находим Xc
    xc = 1.37 / (k ** 0.8)
    # Проверим значение
    if xc >= 0.15:
        xc = 0.15
    elif xc <= qf / E50:
        xc = qf / E50
    return xc


def residual_strength_param_from_xc(xc):
    """Функция находит параметр падения остатичной прочности в зависимости от пика"""

    param = 0.33 - 1.9 * (0.15 - xc)

    return param



def ptgeoexpert_deviator_loading(qf, E50, **kwargs):

    try:
        kwargs["peak"]
    except KeyError:
        kwargs["peak"] = True

    is_defined_fail_strain = False
    try:
        kwargs["fail_strain"]
    except KeyError:
        kwargs["fail_strain"] = define_xc_qf_E(qf, E50)
        is_defined_fail_strain = True


    peak = kwargs.get('peak')
    fail_strain = kwargs.get('fail_strain')

    try:
        kwargs["residual_strength"]
    except KeyError:
        if fail_strain != 0.15:
            kwargs["residual_strength"] = np.random(0.7, 0.8)*qf
        else:
            kwargs["residual_strength"] = 0.95

    residual_strength = kwargs.get('residual_strength')

    if is_defined_fail_strain and fail_strain <= 0.14:
        fail_strain *= np.random.uniform(0.8, 1.05)
        residual_strength *= np.random.uniform(0.8, 1)

        xc_sigma_3 = lambda sigma_3: 1 - 0.0005 * sigma_3
        fail_strain *= xc_sigma_3(200)

        residual_strength *= xc_sigma_3(200)

    residual_strength_param = residual_strength_param_from_xc(fail_strain)*np.random.uniform(0.8, 1.2)

    residual_strength = residual_strength * qf

    deviations_amplitude = [0.04, 0.02, 0.01, True]

    hyp_ratio = 1.
    if E50 <= 70000 and E50 >= 20000:
        hyp_ratio = 0.95 / 68000. * E50 - 0.95 / 34
    elif E50 < 2000:
        hyp_ratio = 0
    elif E50 > 70000:
        hyp_ratio = 0.95





    x, y = curve(qf, E50,
                 xc=fail_strain,
                 x2=residual_strength_param,
                 qf2=residual_strength,
                 qocr=0,
                 amplitude=deviations_amplitude,
                 hyp_ratio=hyp_ratio)
    return x, y


if __name__ == '__main__':
    versions = {
        "Triaxial_Dynamic_Soil_Test": 1.71,
        "Triaxial_Dynamic_Processing": 1.71,
        "Resonance_Column_Siol_Test": 1.1,
        "Resonance_Column_Processing": 1.1
    }
    from matplotlib import rcParams

    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = '12'
    rcParams['axes.edgecolor'] = 'black'
    plt.grid(axis='both', linewidth='0.6')

    x, y = curve(300, 30000, xc=0.03)
    print(residual_strength(0.75, 0.9, 500))
    plt.plot(x, y)
    plt.legend()
    plt.show()

