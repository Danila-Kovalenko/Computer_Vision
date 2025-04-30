import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage import io, color

# ======================= #
#     1. ОПЕРАТОР СОБЕЛЯ  #
# ======================= #
def sobel_edge_detection(image_path):
    """Загрузка изображения и выделение границ оператором Собеля по X и Y."""
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Не удалось загрузить изображение:", image_path)
        return
    
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # Горизонталь
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # Вертикаль
    
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    
    # Показ исходника
    plt.figure()
    plt.title("Собель: Исходное изображение")
    plt.imshow(img_gray)  # не задаём cmap
    plt.axis('off')
    plt.show()
    
    # Показ Sobel X
    plt.figure()
    plt.title("Собель X")
    plt.imshow(abs_sobelx)
    plt.axis('off')
    plt.show()
    
    # Показ Sobel Y
    plt.figure()
    plt.title("Собель Y")
    plt.imshow(abs_sobely)
    plt.axis('off')
    plt.show()
    
    # Показ комбинированного результата
    plt.figure()
    plt.title("Собель: Комбинированный")
    plt.imshow(sobel_combined)
    plt.axis('off')
    plt.show()

# ========================= #
#    2. ОПЕРАТОР КЭННИ      #
# ========================= #
def canny_edge_detection(image_path, low_threshold=100, high_threshold=200):
    """Загрузка изображения и выделение границ оператором Canny с заданными порогами."""
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Не удалось загрузить изображение:", image_path)
        return
    
    edges = cv2.Canny(img_gray, low_threshold, high_threshold)
    
    plt.figure()
    plt.title("Кэнни: Исходное изображение")
    plt.imshow(img_gray)
    plt.axis('off')
    plt.show()
    
    plt.figure()
    plt.title(f"Кэнни: low={low_threshold}, high={high_threshold}")
    plt.imshow(edges)
    plt.axis('off')
    plt.show()

# ==================================== #
# 3. СОБЕЛЬ и КЭННИ на разных типах    #
# ==================================== #
def compare_sobel_canny(images):
    """
    Применить Sobel и Canny для разных изображений.
    images: список путей к файлам (разные типы: пейзаж, объект и т.д.).
    """
    for img_path in images:
        print("========== Обрабатываем:", img_path, "==========")
        sobel_edge_detection(img_path)
        canny_edge_detection(img_path, 100, 200)

# ===================================== #
# 4. СЕГМЕНТАЦИЯ: K-Means на изображении #
# ===================================== #
def kmeans_segmentation(image_path, n_clusters=4, max_iter=300):
    """Применение K-Means к цветному изображению."""
    img = cv2.imread(image_path)
    if img is None:
        print("Не удалось загрузить изображение:", image_path)
        return
    
    # Преобразуем в двумерный массив (N,3), где N = height*width
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=10, random_state=42)
    kmeans.fit(pixel_values)
    
    centers = np.uint8(kmeans.cluster_centers_)
    labels = kmeans.labels_
    
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    
    # Исходник (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title("K-Means: Исходное")
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
    
    # Сегментированное
    segm_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title(f"K-Means: clusters={n_clusters}, max_iter={max_iter}")
    plt.imshow(segm_rgb)
    plt.axis('off')
    plt.show()

# ================================================== #
# 5. СЕГМЕНТАЦИЯ: DBSCAN (перебор eps и min_samples) #
# ================================================== #
def dbscan_segmentation(image_path, eps=5, min_samples=5, 
                        resize_scale=1.0, sample_ratio=1.0):
    """
    Применение DBSCAN к цветному изображению с опциональным уменьшением размера 
    и выборкой части пикселей, чтобы избежать MemoryError.

    :param image_path: путь к изображению
    :param eps: радиус для DBSCAN
    :param min_samples: минимальное число образцов в радиусе
    :param resize_scale: коэффициент для уменьшения изображения (1.0 - без уменьшения)
    :param sample_ratio: доля пикселей, которые будем использовать (1.0 - все пиксели)
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Не удалось загрузить изображение:", image_path)
        return
    
    # 1) При необходимости уменьшим изображение
    if resize_scale < 1.0:
        new_width = int(img.shape[1] * resize_scale)
        new_height = int(img.shape[0] * resize_scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # 2) При необходимости возьмём случайную часть пикселей
    if sample_ratio < 1.0:
        n_total = pixel_values.shape[0]
        n_sample = int(n_total * sample_ratio)
        indices = np.random.choice(n_total, n_sample, replace=False)
        pixel_values = pixel_values[indices]

    # Применяем DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(pixel_values)
    labels = dbscan.labels_
    
    unique_labels = np.unique(labels)
    segmented_data = np.zeros_like(pixel_values)
    
    for label in unique_labels:
        if label == -1:
            # Шум
            segmented_data[labels == label] = [0, 0, 0]
        else:
            mean_val = np.mean(pixel_values[labels == label], axis=0)
            segmented_data[labels == label] = mean_val
    
    # Если мы выборку делали (sample_ratio < 1.0), корректнее отображать только 
    # сегментированные точки, остальные - фон. Либо можно доработать логику, 
    # чтобы потом распространять результат на все пиксели (но это отдельная задача).
    # Для демонстрации сейчас просто покажем результат на уменьшенном/выбранном наборе.
    segmented_image = segmented_data.reshape((-1, 3))
    segmented_image = np.uint8(segmented_image)

    # Чтобы визуализировать корректно, соберём картинку того же размера, 
    # но с чёрным фоном (если делался sample). 
    # Если sample_ratio=1.0, всё совпадет с исходным.
    if resize_scale < 1.0 or sample_ratio < 1.0:
        # Восстанавливаем форму, если брали resize, 
        # но учтём, что при sample это будет "нарезка" пикселей.
        # Для наглядности просто отображаем результат, не пытаясь вставить пиксели в исходную сетку.
        result_img = segmented_image
        # Перейдём в RGB для отображения
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.title(f"DBSCAN (sampled) eps={eps}, min_samples={min_samples}")
        plt.imshow(result_img)
        plt.axis('off')
        plt.show()
    else:
        # Полный размер, никаких подвыборок. Можно отрисовать как раньше:
        segmented_image = segmented_image.reshape(img.shape)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segm_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        
        plt.figure()
        plt.title("DBSCAN: Исходное")
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()
        
        plt.figure()
        plt.title(f"DBSCAN: eps={eps}, min_samples={min_samples}")
        plt.imshow(segm_rgb)
        plt.axis('off')
        plt.show()

# ========================================== #
# 6. Сравнение K-Means, MeanShift, DBSCAN    #
# ========================================== #
def mean_shift_segmentation(image_path):
    """Сегментация с помощью Mean Shift."""
    img = cv2.imread(image_path)
    if img is None:
        print("Не удалось загрузить изображение:", image_path)
        return
    
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Оценка bandwidth
    bandwidth = estimate_bandwidth(pixel_values, quantile=0.2, n_samples=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pixel_values)
    
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    cluster_centers = np.uint8(cluster_centers)
    segmented_data = cluster_centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    
    # Исходное
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title("Mean Shift: Исходное")
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
    
    # Сегментированное
    segm_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title("Mean Shift: Сегментация")
    plt.imshow(segm_rgb)
    plt.axis('off')
    plt.show()

def compare_segmentation_methods(image_path):
    """
    Применить K-Means, Mean Shift, DBSCAN к одному изображению,
    чтобы сравнить результаты.
    """
    # K-Means
    kmeans_segmentation(image_path, n_clusters=4, max_iter=200)
    # Mean Shift
    mean_shift_segmentation(image_path)
    # DBSCAN
    dbscan_segmentation(image_path, eps=5, min_samples=10)

# =============================== #
# 7. АКТИВНЫЙ КОНТУР (SNAKE)      #
# =============================== #
def active_contour_segmentation(image_path, init_shape="circle"):
    """
    Метод активного контура (snake) с разными начальными формами.
    Возможные значения init_shape: circle, ellipse, rectangle, triangle.
    """
    img_color = io.imread(image_path)
    img_gray = color.rgb2gray(img_color)
    img_gray_smooth = gaussian(img_gray, sigma=2.0)
    
    # Создаём набор точек для инициализации
    s = np.linspace(0, 2*np.pi, 400)
    if init_shape == "circle":
        x = 150 + 80*np.cos(s)
        y = 150 + 80*np.sin(s)
    elif init_shape == "ellipse":
        x = 150 + 100*np.cos(s)
        y = 100 + 60*np.sin(s)
    elif init_shape == "rectangle":
        # "развёрнутый" прямоугольник из 400 точек
        side_points = 100
        x = np.concatenate([
            np.linspace(50, 250, side_points),
            np.ones(side_points)*250,
            np.linspace(250, 50, side_points),
            np.ones(side_points)*50
        ])
        y = np.concatenate([
            np.ones(side_points)*50,
            np.linspace(50, 250, side_points),
            np.ones(side_points)*250,
            np.linspace(250, 50, side_points)
        ])
    else:
        # Простой треугольник
        num_points = 100
        p1, p2, p3 = (50, 50), (250, 50), (150, 220)
        side1_x = np.linspace(p1[0], p2[0], num_points)
        side1_y = np.linspace(p1[1], p2[1], num_points)
        side2_x = np.linspace(p2[0], p3[0], num_points)
        side2_y = np.linspace(p2[1], p3[1], num_points)
        side3_x = np.linspace(p3[0], p1[0], num_points)
        side3_y = np.linspace(p3[1], p1[1], num_points)
        x = np.concatenate([side1_x, side2_x, side3_x])
        y = np.concatenate([side1_y, side2_y, side3_y])
    
    init = np.array([y, x]).T  # порядок (row, col)
    
    snake = active_contour(
        img_gray_smooth,
        init,
        alpha=0.02,
        beta=0.2,
        gamma=0.01,
        max_iterations=500
    )
    
    # Показываем результат
    plt.figure()
    plt.title(f"Активный контур (нач. форма: {init_shape})")
    plt.imshow(img_gray)  # исходная в «сером»
    # Отрисуем начальную форму (линия штриховая)
    plt.plot(init[:, 1], init[:, 0], linestyle='dashed')
    # Результирующий контур (линия сплошная)
    plt.plot(snake[:, 1], snake[:, 0])
    plt.axis('off')
    plt.show()

# ========================================== #
# 8. Оптимизация alpha и beta для Snake      #
# ========================================== #
def optimize_active_contour(image_path, alpha_values, beta_values):
    """Перебор параметров alpha и beta на одном изображении с активным контуром."""
    img_color = io.imread(image_path)
    img_gray = color.rgb2gray(img_color)
    img_gray_smooth = gaussian(img_gray, sigma=2.0)
    
    # Инициализация: окружность
    s = np.linspace(0, 2*np.pi, 400)
    x = 150 + 80*np.cos(s)
    y = 150 + 80*np.sin(s)
    init = np.array([y, x]).T
    
    for alpha in alpha_values:
        for beta in beta_values:
            snake = active_contour(
                img_gray_smooth,
                init,
                alpha=alpha,
                beta=beta,
                gamma=0.01,
                max_iterations=500
            )
            
            plt.figure()
            plt.title(f"Snake: alpha={alpha}, beta={beta}")
            plt.imshow(img_gray)
            plt.plot(init[:, 1], init[:, 0], linestyle='dashed')
            plt.plot(snake[:, 1], snake[:, 0])
            plt.axis('off')
            plt.show()

# ==================== #
#       MAIN           #
# ==================== #
def main():
    # 1. Оператор Собеля
    sobel_edge_detection("example1.jpg")
    
    # 2. Оператор Кэнни (попробуем несколько порогов)
    canny_edge_detection("example1.jpg", 50, 150)
    canny_edge_detection("example1.jpg", 100, 200)
    
    # 3. Сравнение Собеля и Кэнни на разных изображениях
    compare_sobel_canny(["landscape.jpg", "object.jpg"])
    
    # 4. Сегментация K-Means
    kmeans_segmentation("example2.jpg", n_clusters=4, max_iter=200)
    
    # 5. Сегментация DBSCAN (разные eps и min_samples), 
    #    плюс возможность уменьшения картинки / sampling.
    dbscan_segmentation("example2.jpg", eps=3, min_samples=5, 
                        resize_scale=0.5, sample_ratio=0.5)
    dbscan_segmentation("example2.jpg", eps=5, min_samples=10, 
                        resize_scale=0.7, sample_ratio=1.0)
    
    # 6. Сравнительный анализ (K-Means, Mean Shift, DBSCAN)
    compare_segmentation_methods("example3.jpg")
    
    # 7. Метод активного контура с разными начальными формами
    for shape in ["circle", "ellipse", "rectangle", "triangle"]:
        active_contour_segmentation("example4.jpg", init_shape=shape)
    
    # 8. Оптимизация alpha и beta для Snake
    alpha_list = [0.01, 0.02]
    beta_list = [0.1, 0.2]
    optimize_active_contour("example4.jpg", alpha_list, beta_list)

if __name__ == "__main__":
    main()
