# importar librerías importantes
import os
import random
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

import cv2


group = None # Cambiar por el número de grupo
# YOUR CODE HERE (Daniela)
group = 19 # poner el número de grupo asignado en Bloque Neón


def determine_classes(group: int) -> list:
    """Función que determina las clases a utilizar en el proyecto.

    Args:
        group (int): Número de grupo.

    Returns:
        list: Lista con los nombres de las clases a utilizar.
    """
    # Se fija la semilla para reproducibilidad
    random.seed(group)
    
    # Se obtienen las categorías y la cantidad de imágenes por categoría
    stats = [('Canis latrans', 1286), ('Castor canadensis', 854), ('Dasypus novemcinctus', 515), ('Didelphis virginiana', 713), 
             ('Lepus californicus', 515), ('Lynx rufus', 854), ('Megaptera novaeangliae', 515), ('Mephitis mephitis', 515), 
             ('Odocoileus hemionus', 1286), ('Odocoileus virginianus', 1795), ('Otospermophilus beecheyi', 854), 
             ('Otospermophilus variegatus', 515), ('Phoca vitulina', 713), ('Procyon lotor', 1286), ('Sciurus carolinensis', 1795), 
             ('Sciurus niger', 1795), ('Sylvilagus audubonii', 515), ('Sylvilagus floridanus', 854), ('Tamias striatus', 854), 
             ('Tamiasciurus hudsonicus', 713), ('Urocyon cinereoargenteus', 515), ('Ursus americanus', 713), ('Vulpes vulpes', 854)]
    
    # Se agrupan las categorías por cantidad de imágenes
    clusters = {}
    for category, num_images in stats:
        if num_images in clusters:
            clusters[num_images].append(category)
        else:
            clusters[num_images] = [category]
    
    # Se selecciona una categoría aleatoria de cada cluster
    categories = []
    for cluster_categories in clusters.values():
        categories += [random.choice(cluster_categories)]
    return categories

assert group is not None, "Por favor, asigna un número de grupo a la variable 'group'."
assert group > 0, "El número de grupo debe ser mayor que 0."
assert group <= 28, "El número de grupo debe ser menor o igual que 28."

categories = determine_classes(group)
print(f"Las classes que usarán en este Mini-proyecto serán:\n{", ".join(categories)}.")


dict_labels = {} # Cambiar por un diccionario con las etiquetas de las clases 

# YOUR CODE HERE (Daniela)
import zipfile

# El dataset viene comprimido en "dataset (1).zip" (en la misma carpeta que este notebook).
# Se usan rutas relativas para que funcione en Windows y Linux.
ZIP_PATH = "dataset (1).zip"
EXTRACT_DIR = "dataset (1)"

# Se descomprime solo si no se ha descomprimido antes
if not os.path.isdir(EXTRACT_DIR):
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(EXTRACT_DIR)

# Se busca la carpeta que contiene train, valid y test
# (el zip puede traer una carpeta interna, p. ej. "dataset (1)/dataset/train")
DATA_DIR = EXTRACT_DIR
for root, dirs, _ in os.walk(EXTRACT_DIR):
    dirs[:] = [d for d in dirs if d != "__MACOSX"]  # Se ignoran carpetas basura de macOS
    if "train" in dirs:
        DATA_DIR = root
        break

# Se ordenan las clases alfabéticamente y se les asigna un ID empezando en 0
dict_labels = {name: i for i, name in enumerate(sorted(categories))}

def get_data(subset: str) -> tuple[list, list]:
  """Función que obtiene las rutas de las imágenes y sus etiquetas.

  Args:
      subset (str): Nombre del subconjunto de datos a utilizar.

  Returns:
      tuple[list, list]: Rutas de las imágenes y sus etiquetas.
  """
  paths = [] # Cambiar por una lista con las rutas de las imágenes
  labels = [] # Cambiar por una lista con las etiquetas de las imágenes
  
  # YOUR CODE HERE (Daniela)
  # Solo se recorren las clases asignadas al grupo
  for name, idx in dict_labels.items():
      # Rutas de todas las imágenes de esta clase en el subset
      class_paths = sorted(glob(os.path.join(DATA_DIR, subset, name, "*")))
      paths += class_paths
      # Cada imagen recibe el ID de su clase
      labels += [idx] * len(class_paths)
  return paths, labels

train_paths, train_labels = get_data("train")
valid_paths, valid_labels = get_data("valid")
test_paths, test_labels = get_data("test")


assert len(train_paths) > 0, "No se ha cargado ningún dato de entrenamiento"
assert len(train_paths) == 5163, "El número de datos de entrenamiento no es correcto"
assert len(valid_paths) > 0, "No se ha cargado ningún dato de validación"
assert len(valid_paths) == 1723, "El número de datos de validación no es correcto"
assert len(test_paths) > 0, "No se ha cargado ningún dato de test"
assert len(test_paths) == 1724, "El número de datos de test no es correcto"

assert type(train_paths[0]) == str, "Los datos de entrenamiento no tienen la forma correcta"
assert type(valid_paths[0]) == str, "Los datos de validación no tienen la forma correcta"
assert type(test_paths[0]) == str, "Los datos de test no tienen la forma correcta"

assert type(train_labels[0]) == int, "Las etiquetas de entrenamiento no tienen la forma correcta"
assert type(valid_labels[0]) == int, "Las etiquetas de validación no tienen la forma correcta"
assert type(test_labels[0]) == int, "Las etiquetas de test no tienen la forma correcta"

assert len(dict_labels) == len(categories), "El diccionario de labels no tiene la longitud correcta"

categories_sorted = sorted(categories)
assert list(dict_labels.keys()) == categories_sorted, "Las llaves del diccionario de labels no son correctas"
assert list(dict_labels.values()) == list(range(len(categories))), "Los valores del diccionario de labels no son correctos"

assert min([dict_labels[train_paths[i].split(os.sep)[-2]] == train_labels[i] for i in range(len(train_paths))]), "Las etiquetas de entrenamiento no coinciden"
assert min([dict_labels[valid_paths[i].split(os.sep)[-2]] == valid_labels[i] for i in range(len(valid_paths))]), "Las etiquetas de validación no coinciden"
assert min([dict_labels[test_paths[i].split(os.sep)[-2]] == test_labels[i] for i in range(len(test_paths))]), "Las etiquetas de test no coinciden"


# YOUR CODE HERE (Daniela)
splits_labels = {"train": train_labels, "valid": valid_labels, "test": test_labels}
splits_paths = {"train": train_paths, "valid": valid_paths, "test": test_paths}

# 1. Cantidad de imágenes por clase en cada split (tabla)
print(f"{'Clase':30s}{'train':>8s}{'valid':>8s}{'test':>8s}{'total':>8s}")
for name, idx in dict_labels.items():
    counts = [labels.count(idx) for labels in splits_labels.values()]
    print(f"{name:30s}" + "".join(f"{c:8d}" for c in counts) + f"{sum(counts):8d}")

# Gráfico de barras con la cantidad de imágenes por clase y split
x = np.arange(len(dict_labels))
width = 0.25
plt.figure(figsize=(12, 5))
for i, (split_name, labels) in enumerate(splits_labels.items()):
    counts = [labels.count(idx) for idx in dict_labels.values()]
    plt.bar(x + i * width, counts, width, label=split_name)
plt.xticks(x + width, list(dict_labels.keys()), rotation=20)
plt.ylabel("Número de imágenes")
plt.title("Imágenes por clase en cada split")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Distribución de los datos entre los splits (porcentaje)
total = sum(len(labels) for labels in splits_labels.values())
for split_name, labels in splits_labels.items():
    print(f"{split_name}: {len(labels)} imágenes ({100 * len(labels) / total:.1f}%)")

# 3. Ejemplos visuales: una imagen aleatoria por clase en cada split
random.seed(0)
fig, ax = plt.subplots(3, len(dict_labels), figsize=(3 * len(dict_labels), 9))
for i, split_name in enumerate(splits_labels):
    for j, (name, idx) in enumerate(dict_labels.items()):
        # Rutas de la clase actual dentro del split
        class_paths = [p for p, l in zip(splits_paths[split_name], splits_labels[split_name]) if l == idx]
        img = cv2.imread(random.choice(class_paths))
        ax[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[i, j].set_title(f"{split_name}\n{name}", fontsize=9)
        ax[i, j].axis("off")
plt.tight_layout()
plt.show()

# 4. Otras estadísticas: tamaños de las imágenes y cuántas están en escala de grises
heights, widths = [], []
n_gray, n_error = 0, 0
for p in train_paths + valid_paths + test_paths:
    img = cv2.imread(p)
    if img is None:  # Imagen que no se pudo leer
        n_error += 1
        continue
    heights.append(img.shape[0])
    widths.append(img.shape[1])
    # Si los 3 canales son iguales, la imagen es en escala de grises
    if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]):
        n_gray += 1

print(f"Imágenes en escala de grises: {n_gray}")
print(f"Imágenes que no se pudieron leer: {n_error}")
print(f"Alto  -> min: {min(heights)}, max: {max(heights)}, promedio: {np.mean(heights):.1f}")
print(f"Ancho -> min: {min(widths)}, max: {max(widths)}, promedio: {np.mean(widths):.1f}")

# Histograma de los tamaños
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(heights, bins=30)
ax[0].set_title("Distribución del alto")
ax[1].hist(widths, bins=30)
ax[1].set_title("Distribución del ancho")
plt.tight_layout()
plt.show()


def processing(Im: np.ndarray, img_size: tuple[int] | None = None, color_space: str | None = None) -> np.ndarray:
    """Función que preprocesa una lista de imágenes.

    Args:
        Im_list (list[np.ndarray]): Lista con las imágenes a preprocesar.
        img_size (tuple[int] | None): Tamaño de la imagen.
        color_space (str | None): Espacio de color a utilizar. Puede ser "rgb", "gray", "hsv" o "lab".

    Returns:
        list[np.ndarray]: Lista con las imágenes preprocesadas.
    """
    
    # YOUR CODE HERE (Daniela)
    # Redimensionar solo si se da un tamaño
    if img_size is not None:
        Im = cv2.resize(Im, img_size)

    # Convertir de BGR al espacio de color pedido (si se da uno)
    if color_space is not None:
        conversions = {
            "rgb": cv2.COLOR_BGR2RGB,
            "gray": cv2.COLOR_BGR2GRAY,
            "hsv": cv2.COLOR_BGR2HSV,
            "lab": cv2.COLOR_BGR2LAB,
        }
        Im = cv2.cvtColor(Im, conversions[color_space])
    
    return Im


Im_list = [cv2.imread(path) for path in train_paths[:5]]
img_size = (224, 224)
color_space = "rgb"
Im_proc = [processing(Im, img_size, color_space) for Im in Im_list]

assert len(Im_proc) == len(Im_list), "El número de imágenes preprocesadas no es correcto"
assert type(Im_proc) == list, "Las imágenes preprocesadas no están en una lista"
assert type(Im_proc[0]) == np.ndarray, "Las imágenes preprocesadas no son arreglos de numpy"
assert np.array(Im_proc).shape[1:] == (224, 224, 3), "Las imágenes preprocesadas no tienen el tamaño correcto"
assert np.array_equal(cv2.cvtColor(cv2.resize(Im_list[0], (224, 224)), cv2.COLOR_BGR2RGB), Im_proc[0]), "Las imágenes preprocesadas no son correctas"

color_space = "gray"
Im_proc = [processing(Im, img_size, color_space) for Im in Im_list]
assert np.array(Im_proc).shape[1:] == (224, 224), "Las imágenes preprocesadas no tienen el tamaño correcto"
assert np.array_equal(cv2.cvtColor(cv2.resize(Im_list[0], (224, 224)), cv2.COLOR_BGR2GRAY), Im_proc[0]), "Las imágenes preprocesadas no son correctas"

color_space = "hsv"
Im_proc = [processing(Im, img_size, color_space) for Im in Im_list]
assert np.array(Im_proc).shape[1:] == (224, 224, 3), "Las imágenes preprocesadas no tienen el tamaño correcto"
assert np.array_equal(cv2.cvtColor(cv2.resize(Im_list[0], (224, 224)), cv2.COLOR_BGR2HSV), Im_proc[0]), "Las imágenes preprocesadas no son correctas"

color_space = "lab"
Im_proc = [processing(Im, img_size, color_space) for Im in Im_list]
assert np.array(Im_proc).shape[1:] == (224, 224, 3), "Las imágenes preprocesadas no tienen el tamaño correcto"
assert np.array_equal(cv2.cvtColor(cv2.resize(Im_list[0], (224, 224)), cv2.COLOR_BGR2LAB), Im_proc[0]), "Las imágenes preprocesadas no son correctas"

Im_proc = [processing(Im, (256, 256)) for Im in Im_list]
assert np.array(Im_proc).shape[1:] == (256, 256, 3), "Las imágenes preprocesadas no tienen el tamaño correcto"
assert np.array_equal(cv2.resize(Im_list[0], (256, 256)), Im_proc[0]), "Las imágenes preprocesadas no son correctas"

Im_proc = [processing(Im, None, "lab") for Im in Im_list]
assert np.array_equal(cv2.cvtColor(Im_list[0], cv2.COLOR_BGR2LAB), Im_proc[0]), "Las imágenes preprocesadas no son correctas"


def data_augmentation(Im: np.ndarray, transformations: list[str], alpha_factor: tuple[float] | None = None, beta_factor: tuple[int] | None = None) -> np.ndarray:
    """Función que realiza data augmentation a una imagen.

    Args:
        Im (np.ndarray): Imagen a transformar.
        transformations (list[str]): Lista de transformaciones a aplicar. Los valores posibles son "flip", "brightness", "contrast" y "brightness and contrast".
        alpha_factor (tuple[float] | None): Factor de ajuste de brillo. Si es None, no se aplica.
        beta_factor (tuple[int] | None): Factor de ajuste de contraste. Si es None, no se aplica.

    Returns:
        np.ndarray: Imagen transformada.
    """
    # YOUR CODE HERE (Daniela)
    # Reflejo respecto al eje y (horizontal)
    if "flip" in transformations:
        Im = cv2.flip(Im, 1)

    # Solo contraste: alpha aleatorio uniforme, sin cambiar el brillo
    if "contrast" in transformations and alpha_factor is not None:
        alpha = random.uniform(alpha_factor[0], alpha_factor[1])
        Im = cv2.convertScaleAbs(Im, alpha=alpha, beta=0)

    # Solo brillo: beta aleatorio uniforme, sin cambiar el contraste
    if "brightness" in transformations and beta_factor is not None:
        beta = random.uniform(beta_factor[0], beta_factor[1])
        Im = cv2.convertScaleAbs(Im, alpha=1, beta=beta)

    # Brillo y contraste al mismo tiempo
    if "brightness and contrast" in transformations and alpha_factor is not None and beta_factor is not None:
        alpha = random.uniform(alpha_factor[0], alpha_factor[1])
        beta = random.uniform(beta_factor[0], beta_factor[1])
        Im = cv2.convertScaleAbs(Im, alpha=alpha, beta=beta)
    return Im


Im = Im_list[0]
transformations = ["flip"]
Im_aug = data_augmentation(Im, transformations)
Im_aug = cv2.cvtColor(Im_aug, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(Im, cv2.COLOR_BGR2RGB))
ax[0].set_title("Imagen original")
ax[0].axis("off")
ax[1].imshow(Im_aug)
ax[1].set_title("Imagen aumentada (solo flip)")
ax[1].axis("off")
plt.show()

assert Im.shape == Im_aug.shape, "La imagen aumentada no tiene el mismo tamaño que la original"
assert np.array_equal(cv2.cvtColor(Im[:, ::-1], cv2.COLOR_BGR2RGB), Im_aug), "La imagen aumentada no es igual a la original"

transformations = ["flip", "contrast"]
random.seed(1) # Para reproducibilidad
Im_aug = data_augmentation(Im, transformations, (1.5, 2.5))
random.seed(1) # Para reproducibilidad
first_way = cv2.convertScaleAbs(Im[:, ::-1], alpha=random.uniform(1.5, 2.5), beta=0)
assert Im_aug.shape == Im.shape, "La imagen aumentada no tiene el mismo tamaño que la original"
assert np.array_equal(first_way, Im_aug), "La imagen aumentada no es igual a la original"

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(Im, cv2.COLOR_BGR2RGB))
ax[0].set_title("Imagen original")
ax[0].axis("off")
ax[1].imshow(cv2.cvtColor(Im_aug, cv2.COLOR_BGR2RGB))
ax[1].set_title("Imagen aumentada (flip y brillo)")
ax[1].axis("off")
plt.show()

transformations = ["flip", "brightness"]
random.seed(1) # Para reproducibilidad
Im_aug = data_augmentation(Im, transformations, None, (20, 80))
random.seed(1) # Para reproducibilidad
first_way = cv2.convertScaleAbs(Im[:, ::-1], alpha=1, beta=random.uniform(20, 80))
assert Im_aug.shape == Im.shape, "La imagen aumentada no tiene el mismo tamaño que la original"
assert np.array_equal(first_way, Im_aug), "La imagen aumentada no es igual a la original"

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(Im, cv2.COLOR_BGR2RGB))
ax[0].set_title("Imagen original")
ax[0].axis("off")
ax[1].imshow(cv2.cvtColor(Im_aug, cv2.COLOR_BGR2RGB))
ax[1].set_title("Imagen aumentada (flip y contraste)")
ax[1].axis("off")
plt.show()

transformations = ["flip", "brightness and contrast"]
random.seed(1) # Para reproducibilidad
Im_aug = data_augmentation(Im, transformations, (1.5, 2.5), (20, 80))
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(Im, cv2.COLOR_BGR2RGB))
ax[0].set_title("Imagen original")
ax[0].axis("off")
ax[1].imshow(cv2.cvtColor(Im_aug, cv2.COLOR_BGR2RGB))
ax[1].set_title("Imagen aumentada (flip, brillo y contraste)")
ax[1].axis("off")
plt.show()

random.seed(1) # Para reproducibilidad
alpha = random.uniform(1.5, 2.5)
beta = random.uniform(20, 80)
first_way = cv2.convertScaleAbs(Im[:, ::-1], alpha=alpha, beta=beta)
assert np.array_equal(first_way, Im_aug), "La imagen aumentada no es igual a la original"


from utils import converter


from utils import *
# converter("Entrega 1")


import shutil

# py_file = converter("Entrega 1")              # devuelve 'Entrega 1.py'
txt_file = py_file.replace(".py", ".txt")
shutil.copy(py_file, txt_file)
print(f"Archivo generado: {txt_file}")
