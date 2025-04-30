import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from skimage.segmentation import (quickshift, 
                                  active_contour)
from skimage.filters import sobel
from skimage import measure, color
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.draw import rectangle_perimeter, polygon_perimeter, ellipse_perimeter



###############################################################################
# 1. ВЫДЕЛЕНИЕ ГРАНИЦ ОПЕРАТОРОМ СОБЕЛЯ
###############################################################################

def sobel_edges_demo(image_path):
    """
    Пример применения оператора Собеля по горизонтали и вертикали.
    image_path: путь к исходному изображению.
    """
    # Читаем в оттенках серого
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise IOError("Не удалось загрузить изображение.")
    
    # Собель по X и по Y
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Для наглядности берём абсолютные значения и приводим к uint8
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    # Объединяем
    sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
    
    # Отображаем
    plt.figure(figsize=(12,4))
    plt.subplot(1,4,1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Исходное")

    plt.subplot(1,4,2)
    plt.imshow(sobel_x_abs, cmap='gray')
    plt.title("Собель по X")

    plt.subplot(1,4,3)
    plt.imshow(sobel_y_abs, cmap='gray')
    plt.title("Собель по Y")

    plt.subplot(1,4,4)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title("Суммарный результат")

    plt.tight_layout()
    plt.show()


###############################################################################
# 2. ВЫДЕЛЕНИЕ ГРАНИЦ ОПЕРАТОРОМ КЭННИ
###############################################################################

def canny_edges_demo(image_path, low_thresh=100, high_thresh=200):
    """
    Пример обнаружения границ с помощью оператора Кэнни.
    low_thresh и high_thresh - пороговые значения для гистерезиса.
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise IOError("Не удалось загрузить изображение.")

    edges = cv2.Canny(img_gray, low_thresh, high_thresh)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Исходное")

    plt.subplot(1,2,2)
    plt.imshow(edges, cmap='gray')
    plt.title(f"Кэнни ({low_thresh}, {high_thresh})")

    plt.tight_layout()
    plt.show()


###############################################################################
# 3. СРАВНЕНИЕ СОБЕЛЯ И КЭННИ НА РАЗНЫХ ТИПАХ ИЗОБРАЖЕНИЙ
###############################################################################

def compare_sobel_canny(image_path, sobel_ksize=3, canny_low=100, canny_high=200):
    """
    Сравнение результатов Собеля и Кэнни на одном изображении.
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5,
                                     cv2.convertScaleAbs(sobel_y), 0.5, 0)
    
    edges_canny = cv2.Canny(img_gray, canny_low, canny_high)

    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Исходное")

    plt.subplot(1,3,2)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title("Собель")

    plt.subplot(1,3,3)
    plt.imshow(edges_canny, cmap='gray')
    plt.title("Кэнни")

    plt.tight_layout()
    plt.show()

    # Здесь можно дать качественный комментарий по итогам:
    # - Собель даёт более «мягкие» границы, выделяя градиенты.
    # - Кэнни чаще формирует более тонкие и чёткие контуры,
    #   но при неправильных порогах может терять важные детали или создавать шум.


###############################################################################
# 4. K-MEANS ДЛЯ СЕГМЕНТАЦИИ
###############################################################################

def kmeans_segmentation_demo(image_path, n_clusters=3, max_iter=300):
    """
    Пример сегментации изображения методом K-Means.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise IOError("Не удалось загрузить изображение.")
    
    # Приводим к виду (h*w, 3)
    h, w, ch = img.shape
    data = img.reshape((-1,3))
    data = np.float32(data)

    # Инициализация и запуск KMeans
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Формируем итог
    segmented = centers[labels].reshape((h, w, ch))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Исходное")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title(f"K-Means (k={n_clusters}, итераций={max_iter})")

    plt.tight_layout()
    plt.show()


###############################################################################
# 5. DBSCAN С РАЗНЫМИ eps И min_samples
###############################################################################

def dbscan_segmentation_demo(image_path, eps_val=5, min_samples_val=10):
    """
    Пример применения DBSCAN для «кластеризации» пикселей.
    Визуальное разделение будет сильно зависеть от параметров.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise IOError("Не удалось загрузить изображение.")

    # Уменьшим размер, чтобы не было слишком тяжело
    # (или можно взять подвыборку)
    scale = 0.5
    new_size = (int(img.shape[1]*scale), int(img.shape[0]*scale))
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    h, w, _ = img_resized.shape
    data = img_resized.reshape((-1, 3))
    data = np.float32(data)

    # DBSCAN
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    labels = dbscan.fit_predict(data)

    # Кол-во кластеров (кроме -1)
    unique_labels = set(labels)
    colors = []
    for lb in unique_labels:
        if lb == -1:
            # "Шум"
            colors.append((0,0,0))  # чёрный
        else:
            # Случайный цвет
            colors.append((np.random.randint(0,255),
                           np.random.randint(0,255),
                           np.random.randint(0,255)))
    
    # Формируем «раскрашенное» изображение
    output = np.zeros_like(data)
    for i, lb in enumerate(labels):
        c = colors[lb] if lb != -1 else (0, 0, 0)
        output[i] = c
    
    output_img = output.reshape((h, w, 3)).astype(np.uint8)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title("Исходное (уменьш.)")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title(f"DBSCAN (eps={eps_val}, min_smpl={min_samples_val})")

    plt.tight_layout()
    plt.show()


###############################################################################
# 6. СРАВНИТЕЛЬНЫЙ АНАЛИЗ (K-MEANS, MEAN SHIFT, DBSCAN)
###############################################################################

from skimage.segmentation import (quickshift, felzenszwalb)
# Или можно использовать встроенный Mean Shift из sklearn.cluster (с ограничениями)

def compare_segmentation_methods(image_path):
    """
    Сравнение K-Means, Mean Shift (пример с quickshift) и DBSCAN.
    Для Mean Shift можно показать quickshift из skimage,
    которая работает по похожим принципам кластеризации с учётом пространственного
    соседства и интенсивности.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise IOError("Не удалось загрузить изображение.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Подготовим для KMeans
    h, w, ch = img.shape
    data = img.reshape((-1,3))
    data = np.float32(data)

    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(data)
    segmented_km = kmeans.cluster_centers_.astype(np.uint8)[labels_km].reshape((h, w, ch))

    # Mean Shift (для наглядности можно использовать quickshift)
    # quickshift возвращает метки
    segments_qs = quickshift(img_rgb, kernel_size=3, max_dist=6, ratio=0.5)
    # Возвращаем в RGB для визуализации
    qs_boundaries = color.label2rgb(segments_qs, img_rgb, kind='avg')

    # DBSCAN
    # (Для наглядности уменьшим размер)
    scale = 0.5
    new_size = (int(w*scale), int(h*scale))
    img_res = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)
    data_res = img_res.reshape((-1,3))
    db = DBSCAN(eps=5, min_samples=10)
    labels_db = db.fit_predict(data_res)
    unique_labels = set(labels_db)
    colors = []
    for lb in unique_labels:
        if lb == -1:
            colors.append((0,0,0))
        else:
            colors.append((np.random.randint(0,255),
                           np.random.randint(0,255),
                           np.random.randint(0,255)))
    output_db = np.zeros_like(data_res, dtype=np.uint8)
    for i, lb in enumerate(labels_db):
        output_db[i] = colors[lb]
    dbscan_img = output_db.reshape(img_res.shape)

    # Отображаем
    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.imshow(img_rgb)
    plt.title("Исходное")

    plt.subplot(2,2,2)
    plt.imshow(cv2.cvtColor(segmented_km, cv2.COLOR_BGR2RGB))
    plt.title("K-Means")

    plt.subplot(2,2,3)
    plt.imshow(qs_boundaries)
    plt.title("Mean Shift (quickshift)")

    plt.subplot(2,2,4)
    plt.imshow(dbscan_img)
    plt.title("DBSCAN (уменьшенное изображение)")

    plt.tight_layout()
    plt.show()


###############################################################################
# 7. МЕТОД АКТИВНОГО КОНТУРА (SNAKES) С РАЗНЫМИ НАЧАЛЬНЫМИ ФОРМАМИ
###############################################################################

from skimage.draw import polygon
from skimage import img_as_float

def active_contour_demo(image_path):
    """
    Пример применения метода активного контура с разными начальными формами:
    прямоугольник, треугольник и эллипс.
    """
    img = imread(image_path)
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = rgb2gray(img)
    else:
        gray = img_as_float(img)
    
    # Преобразуем в float, если нужно
    gray = img_as_float(gray)

    # Форма снимка
    h, w = gray.shape

    # 1) Прямоугольник
    # Зададим координаты периметра
    rect_start = (h//4, w//4)
    rect_end = (3*h//4, 3*w//4)
    rr, cc = rectangle_perimeter(rect_start, end=rect_end)
    init_rect = np.array([rr, cc]).T

    snake_rect = active_contour(gray, 
                                init_rect, 
                                alpha=0.015,  # пример значения
                                beta=10,      # пример значения
                                iterations=200)

    # 2) Треугольник (произвольный)
    coords_triangle = np.array([(h*0.3, w*0.3), 
                                (h*0.3, w*0.7), 
                                (h*0.7, w*0.5)])
    # Получаем контур
    rr_t, cc_t = polygon(coords_triangle[:,0], coords_triangle[:,1])
    init_triangle = np.array([rr_t, cc_t]).T

    snake_triangle = active_contour(gray,
                                    init_triangle,
                                    alpha=0.015,
                                    beta=10,
                                    iterations=200)

    # 3) Эллипс
    center = (h//2, w//2)
    r_radius = h//4
    c_radius = w//6
    rr_e, cc_e = ellipse_perimeter(center[0], center[1], r_radius, c_radius)
    init_ellipse = np.array([rr_e, cc_e]).T

    snake_ellipse = active_contour(gray,
                                   init_ellipse,
                                   alpha=0.015,
                                   beta=10,
                                   iterations=200)

    # Отображаем результаты
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    ax = axes.ravel()

    ax[0].imshow(gray, cmap='gray')
    ax[0].plot(init_rect[:,1], init_rect[:,0], '--r', lw=2)
    ax[0].set_title("Начальная (прямоугольник)")

    ax[1].imshow(gray, cmap='gray')
    ax[1].plot(snake_rect[:,1], snake_rect[:,0], '-b', lw=2)
    ax[1].set_title("Результат (прямоугольник)")

    ax[2].imshow(gray, cmap='gray')
    ax[2].plot(init_triangle[:,1], init_triangle[:,0], '--r', lw=2)
    ax[2].set_title("Начальная (треугольник)")

    ax[3].imshow(gray, cmap='gray')
    ax[3].plot(snake_triangle[:,1], snake_triangle[:,0], '-b', lw=2)
    ax[3].set_title("Результат (треугольник)")

    ax[4].imshow(gray, cmap='gray')
    ax[4].plot(init_ellipse[:,1], init_ellipse[:,0], '--r', lw=2)
    ax[4].set_title("Начальная (эллипс)")

    ax[5].imshow(gray, cmap='gray')
    ax[5].plot(snake_ellipse[:,1], snake_ellipse[:,0], '-b', lw=2)
    ax[5].set_title("Результат (эллипс)")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.show()


###############################################################################
# 8. ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ALPHA И BETA
###############################################################################

def optimize_active_contour_params(image_path, alpha_list, beta_list):
    """
    Пример перебора разных alpha и beta для одного и того же изображения
    в методе активного контура.
    Показываем, как меняется итоговый контур.
    """
    img = imread(image_path)
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = rgb2gray(img)
    else:
        gray = img_as_float(img)

    h, w = gray.shape
    # Простейшая стартовая фигура (к примеру, эллипс в центре)
    center = (h//2, w//2)
    r_radius = h//4
    c_radius = w//6
    rr_e, cc_e = ellipse_perimeter(center[0], center[1], r_radius, c_radius)
    init_snake = np.array([rr_e, cc_e]).T

    fig, axes = plt.subplots(len(alpha_list), len(beta_list), figsize=(12, 12))
    for i, alpha_val in enumerate(alpha_list):
        for j, beta_val in enumerate(beta_list):
            snake_result = active_contour(gray,
                                          init_snake,
                                          alpha=alpha_val,
                                          beta=beta_val,
                                          iterations=200)
            ax = axes[i, j]
            ax.imshow(gray, cmap='gray')
            ax.plot(init_snake[:,1], init_snake[:,0], '--r', lw=1)
            ax.plot(snake_result[:,1], snake_result[:,0], '-b', lw=2)
            ax.set_title(f"alpha={alpha_val}, beta={beta_val}")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

