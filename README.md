# Computer Vision – Lern- und Übungsrepository

Dieses Repository ist eine **Sammlung von Python-Skripten** zu den Grundlagen von Machine Learning und Computer Vision.  
Der Schwerpunkt liegt auf praktischen Experimenten mit:

- Perzeptronen und kleinen neuronalen Netzen,
- Bildvorverarbeitung und Kantenextraktion,
- Bildsegmentierung mit klassischen Verfahren,
- interaktiven/visuellen Demonstrationen mit Matplotlib und GUI-Elementen.

## Inhalt des Repositories

- `main.py` – Vergleich zwischen einfachem Perzeptron und MLP auf dem XOR-Problem (PyTorch).
- `2.py` – Klassisches Perzeptron-Training für OR-Logik mit NumPy.
- `3.py` – Manuell implementiertes neuronales Netz (u. a. ReLU, Softmax, Backpropagation, Dropout).
- `4.py` – Segmentierungs- und Binärisierungswerkzeuge inklusive GUI-Komponente (`SimpleSegmentationGUI`).
- `5.py` – Demos für Sobel/Canny, K-Means, DBSCAN und aktive Konturen.
- `5no1.py` – Alternative/erweiterte Version klassischer CV-Experimente (inkl. Mean Shift und Parameteroptimierung).
- Bilddateien (`example*.jpg`, `landscape.jpg`, `object.jpg`) als Eingabedaten für die Skripte.

## Projektidee in einem Satz

Das Projekt zeigt, wie man mit **klassischen Computer-Vision-Methoden** und **einfachen neuronalen Modellen** schrittweise von Logikbeispielen (OR/XOR) zu realen Bildanalysen übergeht.

## Verwendete Technologien

- **Python 3**
- **NumPy**
- **PyTorch**
- **OpenCV (cv2)**
- **scikit-learn**
- **scikit-image**
- **Matplotlib**
- **Pillow / Tkinter** (für GUI- bzw. Anzeige-Funktionen)

## Schnellstart

1. Abhängigkeiten installieren (Beispiel):

```bash
pip install numpy matplotlib torch torchvision opencv-python scikit-learn scikit-image pillow
```

2. Ein Skript starten, z. B.:

```bash
python main.py
```

oder:

```bash
python 5.py
```

> Hinweis: Einige Skripte öffnen Diagramme/Fenster interaktiv. In Headless-Umgebungen kann die Darstellung eingeschränkt sein.

## Für wen ist dieses Repository gedacht?

- Studierende / Einsteiger in Computer Vision,
- Personen, die klassische Segmentierungsverfahren direkt am Bild testen wollen,
- alle, die den Unterschied zwischen linearen und nichtlinearen Lernproblemen (OR vs. XOR) praktisch sehen möchten.

## Lizenz

Dieses Projekt steht unter der in `LICENSE` angegebenen Lizenz.
