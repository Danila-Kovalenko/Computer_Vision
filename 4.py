import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk
from collections import deque

# =================================================================
# 1. ФУНКЦИИ ДЛЯ ЗАГРУЗКИ И ПРЕДОБРАБОТКИ ИЗОБРАЖЕНИЯ
# =================================================================

def load_grayscale_image(path):
    """
    Загружаем изображение в оттенках серого.
    Возвращаем ndarray (h, w) типа uint8.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Не удалось загрузить файл: {path}")
    return img

# =================================================================
# 2. БИНАРИЗАЦИЯ (cv2.threshold, cv2.adaptiveThreshold)
# =================================================================

def binarize_with_threshold(img, thresh_value=128):
    """
    Пример простой пороговой бинаризации (cv2.threshold).
    Возвращает бинаризованное изображение.
    """
    # cv2.THRESH_BINARY – пиксели, яркость которых >= thresh_value, станут белыми, иначе чёрными.
    # Возвращает: ret (сам thresh_value) и бинаризированное изображение
    _, binary = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)
    return binary

def binarize_with_adaptive_threshold(img, block_size=11, C=2):
    """
    Адаптивная бинаризация (cv2.adaptiveThreshold).
    block_size – размер области для адаптивного вычисления порога.
    C – константа, вычитаемая из среднего или взвешенного среднего.
    """
    # cv2.ADAPTIVE_THRESH_MEAN_C – среднее по блоку
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C – гауссово взвешенное
    adaptive_bin = cv2.adaptiveThreshold(
        img, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        C
    )
    return adaptive_bin

def compare_binarization_methods(img, thresh_value=128, block_size=11, C=2):
    """
    Показываем сравнение двух методов (простого порога и адаптивного).
    """
    bin_simple = binarize_with_threshold(img, thresh_value)
    bin_adaptive = binarize_with_adaptive_threshold(img, block_size, C)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Исходное")

    plt.subplot(1, 3, 2)
    plt.imshow(bin_simple, cmap='gray')
    plt.title(f"cv2.threshold (порог={thresh_value})")

    plt.subplot(1, 3, 3)
    plt.imshow(bin_adaptive, cmap='gray')
    plt.title(f"cv2.adaptiveThreshold (block={block_size}, C={C})")

    plt.tight_layout()
    plt.show()

# =================================================================
# 3. РЕАЛИЗАЦИЯ ФУНКЦИИ REGION GROWING
# =================================================================
def region_growing(img, seed_point, threshold=10):
    """
    Реализуем простейший Region Growing.
    img – uint8, оттенки серого.
    seed_point – (y, x), начальный пиксель.
    threshold – порог по разнице интенсивности.
    
    Возвращаем бинарную маску (0 или 255), где 255 – область, 
    принадлежащая региону роста.
    """
    h, w = img.shape
    output = np.zeros((h, w), dtype=np.uint8)
    
    seed_y, seed_x = seed_point
    seed_val = img[seed_y, seed_x]
    
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    queue.append((seed_y, seed_x))
    visited[seed_y, seed_x] = True
    
    while queue:
        cy, cx = queue.popleft()
        output[cy, cx] = 255  # или 1, если хотите чёрно-белую
    
        # Соседи (4-связность или 8-связность)
        for ny, nx in [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]:
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                # Проверяем разницу по интенсивности
                if abs(int(img[ny, nx]) - int(seed_val)) <= threshold:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    
    return output

# =================================================================
# 4. SPLIT-AND-MERGE
# =================================================================
def split_and_merge(img, max_std=10, min_size=32):
    """
    Упрощённый алгоритм разбиения и объединения (quadtree).
    - Если стандартное отклонение яркости в блоке > max_std и блок больше min_size,
      то делим блок на 4 квадранта.
    - Иначе блок объявляем конечным регионом.
    Возвращает матрицу меток (labels), где у каждого сегмента своя метка.
    
    Внимание: алгоритм рекурсивный, может работать медленно на больших изображениях.
    """
    h, w = img.shape
    # Будем labels хранить отдельно
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 1
    
    def split_merge_recursive(y, x, height, width, label):
        nonlocal current_label
        
        region = img[y:y+height, x:x+width]
        region_std = np.std(region)
        
        if region_std > max_std and height > min_size and width > min_size:
            # Разбиваем
            half_h = height // 2
            half_w = width // 2
            split_merge_recursive(y, x, half_h, half_w, label)
            split_merge_recursive(y, x + half_w, half_h, width - half_w, label)
            split_merge_recursive(y + half_h, x, height - half_h, half_w, label)
            split_merge_recursive(y + half_h, x + half_w, height - half_h, width - half_w, label)
        else:
            # Заполняем labels текущей меткой
            labels[y:y+height, x:x+width] = current_label
            current_label += 1
    
    split_merge_recursive(0, 0, h, w, current_label)
    return labels

def visualize_split_and_merge(img, labels):
    """
    Визуализация результатов split-and-merge
    """
    # Для красоты окрасим разные метки в разные цвета
    # (Хак: нормируем метки и применяем колormap)
    max_label = labels.max()
    # Нормируем в диапазон [0..1]
    normalized = (labels.astype(np.float32) / max_label) if max_label != 0 else labels
    colored = cv2.applyColorMap((normalized*255).astype(np.uint8), cv2.COLORMAP_HSV)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Исходное изображение")
    
    plt.subplot(1,2,2)
    plt.imshow(colored[...,::-1])  # BGR->RGB
    plt.title("Split-and-Merge результат")
    plt.show()

# =================================================================
# 5. WATERSHED SEGMENTATION
# =================================================================
def watershed_segmentation(img, distance_radius=3, min_distance_val=5):
    """
    Пример сегментации методом Watershed.
    Используем морфологические операции для нахождения "sure foreground".
    distance_radius – размер структуры при морфологическом расширении/эрозии.
    min_distance_val – минимальное расстояние между пиками в distance map для peak_local_max.
    """
    # Сначала бинаризуем (можно адаптивно)
    bin_img = binarize_with_adaptive_threshold(img, block_size=21, C=5)

    # Морфологические операции, чтобы выделить фон и передний план
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance_radius, distance_radius))
    
    # sure background (opening -> dilation)
    sure_bg = cv2.dilate(cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel), kernel, iterations=2)

    # distance transform
    dist_transform = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    # Найдём локальные максимумы
    local_max = peak_local_max(dist_transform, indices=False, min_distance=min_distance_val, labels=bin_img)
    
    # Маркируем их
    markers = cv2.connectedComponents(local_max.astype(np.uint8))[1]
    # Watershed работает с цветным (3-канальным) изображением
    # Но можно его "символически" применить на псевдо-RGB
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Применяем watershed
    # markers < 0 после watershed -> границы
    markers_watershed = cv2.watershed(color_img, markers.copy())
    
    # Разукрасим метки
    max_m = markers_watershed.max()
    normalized = (markers_watershed.astype(np.float32) / max_m) if max_m!=0 else markers_watershed
    colored = cv2.applyColorMap((normalized*255).astype(np.uint8), cv2.COLORMAP_JET)
    
    return bin_img, markers_watershed, colored

# =================================================================
# 6. ПРОСТОЕ ГРАФИЧЕСКОЕ ПРИЛОЖЕНИЕ (tkinter)
# =================================================================

class SimpleSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Пример приложения сегментации")
        
        # Переменные
        self.image_path = None
        self.original_image = None
        
        # Интерфейс
        self.btn_load = tk.Button(root, text="Загрузить изображение", command=self.load_image)
        self.btn_load.pack(pady=5)
        
        # Выбор метода
        self.method_label = tk.Label(root, text="Метод сегментации / бинаризации:")
        self.method_label.pack()
        
        self.method_var = tk.StringVar(value="threshold")
        self.methods_combo = ttk.Combobox(root, textvariable=self.method_var, 
                                          values=["threshold", "adaptiveThreshold", 
                                                  "regionGrowing", "splitMerge", "watershed"])
        self.methods_combo.pack(pady=5)
        
        # Параметры
        frame_params = tk.Frame(root)
        frame_params.pack()
        
        tk.Label(frame_params, text="Порог (threshold):").grid(row=0, column=0, sticky='e')
        self.entry_threshold = tk.Entry(frame_params)
        self.entry_threshold.insert(0, "128")
        self.entry_threshold.grid(row=0, column=1)
        
        tk.Label(frame_params, text="Block Size (адапт.):").grid(row=1, column=0, sticky='e')
        self.entry_block_size = tk.Entry(frame_params)
        self.entry_block_size.insert(0, "11")
        self.entry_block_size.grid(row=1, column=1)
        
        tk.Label(frame_params, text="C (адапт.):").grid(row=2, column=0, sticky='e')
        self.entry_C = tk.Entry(frame_params)
        self.entry_C.insert(0, "2")
        self.entry_C.grid(row=2, column=1)
        
        tk.Label(frame_params, text="RegGrow порог:").grid(row=3, column=0, sticky='e')
        self.entry_rg_threshold = tk.Entry(frame_params)
        self.entry_rg_threshold.insert(0, "15")
        self.entry_rg_threshold.grid(row=3, column=1)
        
        tk.Label(frame_params, text="Seed Y,X (RegGrow):").grid(row=4, column=0, sticky='e')
        self.entry_rg_seed = tk.Entry(frame_params)
        self.entry_rg_seed.insert(0, "50,50")
        self.entry_rg_seed.grid(row=4, column=1)
        
        tk.Label(frame_params, text="SplitMerge max_std:").grid(row=5, column=0, sticky='e')
        self.entry_sm_maxstd = tk.Entry(frame_params)
        self.entry_sm_maxstd.insert(0, "10")
        self.entry_sm_maxstd.grid(row=5, column=1)
        
        tk.Label(frame_params, text="SplitMerge min_size:").grid(row=6, column=0, sticky='e')
        self.entry_sm_minsize = tk.Entry(frame_params)
        self.entry_sm_minsize.insert(0, "32")
        self.entry_sm_minsize.grid(row=6, column=1)
        
        tk.Label(frame_params, text="Watershed distance_radius:").grid(row=7, column=0, sticky='e')
        self.entry_ws_radius = tk.Entry(frame_params)
        self.entry_ws_radius.insert(0, "3")
        self.entry_ws_radius.grid(row=7, column=1)
        
        tk.Label(frame_params, text="Watershed min_distance:").grid(row=8, column=0, sticky='e')
        self.entry_ws_mindist = tk.Entry(frame_params)
        self.entry_ws_mindist.insert(0, "5")
        self.entry_ws_mindist.grid(row=8, column=1)
        
        # Кнопка "Выполнить"
        self.btn_run = tk.Button(root, text="Выполнить", command=self.run_method)
        self.btn_run.pack(pady=10)
        
        # Метка для отображения результата
        self.result_label = tk.Label(root)
        self.result_label.pack()

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение", 
            filetypes=[("Изображения", "*.png *.jpg *.bmp *.jpeg *.pgm *.ppm"), ("Все файлы", "*.*")]
        )
        if path:
            self.image_path = path
            # Загрузим
            self.original_image = load_grayscale_image(self.image_path)
            # Отобразим миниатюру
            self.show_image_in_label(self.original_image, self.result_label)

    def show_image_in_label(self, img_gray, label_widget):
        # img_gray – np.array (h,w), grayscale
        # Конвертируем в RGB для PIL
        h, w = img_gray.shape
        # Создадим 3-канальную копию
        display_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(display_img)
        pil_img = pil_img.resize((300, 300))  # масштабируем для окна
        tk_img = ImageTk.PhotoImage(image=pil_img)
        label_widget.configure(image=tk_img)
        label_widget.image = tk_img  # чтобы объект не удалился сборщиком мусора

    def run_method(self):
        if self.original_image is None:
            print("Сначала загрузите изображение.")
            return
        
        method = self.method_var.get()
        img = self.original_image.copy()
        
        if method == "threshold":
            tval = int(self.entry_threshold.get())
            result = binarize_with_threshold(img, tval)
            self.show_image_in_label(result, self.result_label)

        elif method == "adaptiveThreshold":
            block_size = int(self.entry_block_size.get())
            c_val = int(self.entry_C.get())
            result = binarize_with_adaptive_threshold(img, block_size, c_val)
            self.show_image_in_label(result, self.result_label)

        elif method == "regionGrowing":
            rg_thresh = int(self.entry_rg_threshold.get())
            seed_txt = self.entry_rg_seed.get()
            seed_y, seed_x = [int(v) for v in seed_txt.split(',')]
            result = region_growing(img, (seed_y, seed_x), rg_thresh)
            self.show_image_in_label(result, self.result_label)

        elif method == "splitMerge":
            sm_max_std = float(self.entry_sm_maxstd.get())
            sm_min_size = int(self.entry_sm_minsize.get())
            labels = split_and_merge(img, max_std=sm_max_std, min_size=sm_min_size)
            # Визуализируем в отдельном окошке matplotlib
            visualize_split_and_merge(img, labels)

        elif method == "watershed":
            dist_radius = int(self.entry_ws_radius.get())
            min_dist = int(self.entry_ws_mindist.get())
            _, _, colored = watershed_segmentation(img, distance_radius=dist_radius, min_distance_val=min_dist)
            # colored – BGR, приведём к GRAY или оставим цветным
            # Для отображения в Label (PIL) – нужно RGB
            colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(colored_rgb)
            pil_img = pil_img.resize((300, 300))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.result_label.configure(image=tk_img)
            self.result_label.image = tk_img

# =================================================================
# ЗАПУСК ГЛАВНОГО ОКНА
# =================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleSegmentationGUI(root)
    root.mainloop()
