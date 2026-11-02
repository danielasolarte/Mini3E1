# importar librerias
import os
import random
import numpy as np
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt

import cv2
from skimage.feature import hog

from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support


group = None # Cambiar por el número de grupo
# YOUR CODE HERE (Daniela)
group = 19 # Número de grupo asignado (el mismo de las entregas 1 y 2)


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
# Carpeta con los datos (train, valid y test), al mismo nivel que este notebook
DATA_DIR = "dataset"

# Se ordenan las clases alfabéticamente y se les asigna un ID empezando en 0
dict_labels = {name: i for i, name in enumerate(sorted(categories))}

def get_data(subset: str) -> tuple[list[str], list[int]]:
  """Función que obtiene las rutas de las imágenes y sus etiquetas.

  Args:
      subset (str): Nombre del subconjunto de datos a utilizar.

  Returns:
      tuple[list, list]: Rutas de las imágenes y sus etiquetas.
  """
  paths = []
  labels = []
  
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


def color_hist(Im: np.ndarray, type_h:str, bins: int) -> np.ndarray:
    """Función que calcula el histograma de color conjunto o concatenado de una imagen

    Args:
        Im (numpy.ndarray): Imagen a la que calcular el histograma
        type_h (str): Tipo de histograma a calcular, puede ser "concat" o "joint"
        bins (int): Cantidad de bins que usar en el histograma. En el caso de concat el número de bins finale será 3*bins
        En el caso de Joint será bins^3
        
    Returns:YOUR CO
        numpy.ndarray: Histograma de color normalizado
    """
    # YOUR CODE HERE (Daniela)
    # Misma función de la entrega 2: el rango de los bins va del mínimo al máximo de cada canal
    if type_h == "concat":
        # Un histograma por cada canal y luego se ponen uno detrás del otro
        h = np.concatenate([np.histogram(Im[:, :, c], bins=bins)[0] for c in range(3)])
    elif type_h == "joint":
        # Un solo histograma 3D con todas las combinaciones de los 3 canales, luego se aplana
        h = np.histogramdd(Im.reshape(-1, 3), bins=bins)[0].flatten()

    # Se normaliza para que la suma total sea 1
    h = h / np.sum(h)
    return h


img=processing(cv2.imread(os.path.join(".", "dataset", "train", "Canis latrans", "f494fab00ebb2d87c8d3bf5215560d54.jpg")),  (256, 256), 'rgb')
concat_h=color_hist(img,'concat',10)
joint_h=color_hist(img,'joint',10)


fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].hist(list(range(len(concat_h))), list(range(len(concat_h - 1))), weights = concat_h)
ax[0].set_title('Histograma concatenado')
ax[1].hist(list(range(len(joint_h))), list(range(len(joint_h - 1))), weights = joint_h)
ax[1].set_title('Histograma conjunto')
ax[2].imshow(img)
ax[2].axis('off')
ax[2].set_title('Imagen original')

plt.tight_layout()

assert np.isclose(np.sum(concat_h),1,0.1), 'Su histograma concatenado no esta normalizado'
assert np.isclose(np.sum(joint_h),1,0.1), 'Su histograma conjunto no esta normalizado'
assert len(concat_h.shape)==1 and len(joint_h.shape)==1, 'Su histograma debe tener una unica dimension'
assert np.isclose(len(concat_h)/3,len(joint_h)**(1/3),0.01),' Sus histogramas no tienen la cantidad de bins que deberían'
assert np.isclose(np.std(concat_h),0.025,0.05), 'Su histograma de color concatenado es erroneo'
assert np.isclose(np.std(joint_h),0.0063,0.05), 'Su histograma de color conjunto es erroneo'


def Color_pyramid(Im: np.ndarray, patch_sizes: list[tuple[int]], type_h: str, bins:int) -> np.ndarray:
    """Función que calcula el histograma de color de una imagen por parches

    Args:
        Im (numpy.ndarray): Imagen de entrada
        patch_sizes (list[tuple[int]]): Tupla tamaño de los parches sobre el que calcular cada histograma
        type_h (str): Puede ser "concat" o "joint"
        bins (int): Cantidad de bins que usar en el histograma

    Returns:
        numpy.ndarray: Histograma de color normalizado
    """
    # YOUR CODE HERE (Daniela)
    hists = []
    # Se recorre cada nivel de la pirámide con su tamaño de parche
    for (ph, pw) in patch_sizes:
        # Parches de izquierda a derecha y de arriba a abajo
        for y in range(0, Im.shape[0], ph):
            for x in range(0, Im.shape[1], pw):
                patch = Im[y:y + ph, x:x + pw]
                # Histograma normalizado del parche
                hists.append(color_hist(patch, type_h, bins))

    # Se concatenan todos los histogramas (el resultado no queda normalizado)
    final_histogram = np.concatenate(hists)
    return final_histogram


img=processing(cv2.imread(os.path.join(".", "dataset", "train", "Canis latrans", "f494fab00ebb2d87c8d3bf5215560d54.jpg")),  (256, 256), 'rgb')
h_pyramid_cat=Color_pyramid(img,[(256, 256), (128, 128), (64, 64)],'concat',10)
h_pyramid_joint=Color_pyramid(img,[(256, 256), (128, 128), (64, 64)],'joint',5)

fig,ax=plt.subplots(2,2,figsize=(10,10))
ax[0,0].imshow(img)
ax[0,0].axis('off')
ax[0,0].set_title('Imagen')
ax[0,1].hist(list(range(len(h_pyramid_cat))), list(range(len(h_pyramid_cat - 1))), weights = h_pyramid_cat)
ax[0,1].set_title('Histograma de piramide')

assert len(h_pyramid_cat)==30*(1+4+16), 'La longitud del descriptor final debe ser igual a la longitud del histograma concatenado multiplicado por la cantidad de cuadrantes dentro de la imagen'
assert np.isclose(np.sum(h_pyramid_cat),(1+4+16),0.01), 'En total se tienen 64 histogramas normalizados juntos, la suma de todo debe ser 64'
assert len(h_pyramid_joint)==5**3*(1+4+16), 'La longitud del descriptor final debe ser igual a la longitud del histograma conjunto multiplicado por la cantidad de cuadrantes dentro de la imagen'
assert np.isclose(np.sum(h_pyramid_joint),(1+4+16),0.01), 'En total se tienen 64 histogramas normalizados juntos, la suma de todo debe ser 64'
a=color_hist(img[64:64+64,64:64+64,:],'concat',10)
K=10
b=h_pyramid_cat[K*30:K*30+30]

ax[1,0].imshow(img[64:64+64,64:64+64,:])
ax[1,0].axis('off')
ax[1,0].set_title('Recorte')
ax[1,1].hist(list(range(len(h_pyramid_cat[K*30:K*30+30]))), list(range(len(h_pyramid_cat[K*30:K*30+30] - 1))), weights = h_pyramid_cat[K*30:K*30+30])
ax[1,1].set_title('Histograma de recorte')

assert np.array_equal(a,b), 'El histograma concatenado de el cuadrante 17 de la imagen debe ser igual la parte 17 del histograma en piramide'


hist_hog='' #Histograma de gradientes de la imagen
hog_map='' #Imagen de mapa de histogramas de gradientes
# YOUR CODE HERE (Daniela)
# No hay gatos en nuestras clases, así que se usa la misma imagen de ejemplo de Canis latrans
img_gray = processing(cv2.imread(os.path.join(".", "dataset", "train", "Canis latrans", "f494fab00ebb2d87c8d3bf5215560d54.jpg")), (256, 256), "gray")

# HOG: 9 orientaciones, celdas de 32x32 píxeles y bloques de 2x2 celdas
hist_hog, hog_map = hog(img_gray, orientations=9, pixels_per_cell=(32, 32),
                        cells_per_block=(2, 2), visualize=True)

# Visualización de la imagen, el mapa de gradientes y el histograma
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img_gray, cmap="gray")
ax[0].axis("off")
ax[0].set_title("Imagen en escala de grises")
ax[1].imshow(hog_map, cmap="gray")
ax[1].axis("off")
ax[1].set_title("Mapa HOG")
ax[2].plot(hist_hog)
ax[2].set_title(f"Descriptor HOG ({len(hist_hog)} dimensiones)")
plt.tight_layout()


train_images_num='' #cantidad de imagens de entrenamiento
train_concat_color_pyramid_desc=[] #Descriptor de imagenes con histogramas concatenados de color en piramide
train_hog_desc=[] #Descriptor de imagenes con histogramas de gradientes orientados

valid_concat_color_pyramid_desc=[] #Descriptor de imagenes con histogramas concatenados de color en piramide
valid_hog_desc=[] #Descriptor de imagenes con histogramas de gradientes orientados

# YOUR CODE HERE (Daniela)
img_size = (256, 256)                                # Tamaño estándar de las imágenes
patch_sizes = [(256, 256), (128, 128), (64, 64)]     # 3 niveles de la pirámide (1 + 4 + 16 parches)
color_space = "hsv"                                  # Espacio de color (el mejor de la entrega 2)
bins = 10                                            # Bins por canal -> 30 * 21 = 630 dimensiones
hog_params = dict(orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2))  # -> 1764 dimensiones

# Se usan todas las imágenes de entrenamiento
train_images_num = len(train_paths)

def get_descriptors(path):
    """Calcula el descriptor de color en pirámide y el HOG de una imagen."""
    Im = cv2.imread(path)
    color_desc = Color_pyramid(processing(Im, img_size, color_space), patch_sizes, "concat", bins)
    hog_desc = hog(processing(Im, img_size, "gray"), **hog_params)
    return color_desc, hog_desc

# Descriptores de entrenamiento (en el mismo orden que las etiquetas)
for path in tqdm(train_paths[:train_images_num], desc="Train"):
    color_desc, hog_desc = get_descriptors(path)
    train_concat_color_pyramid_desc.append(color_desc)
    train_hog_desc.append(hog_desc)

# Descriptores de validación, con los mismos parámetros
for path in tqdm(valid_paths, desc="Valid"):
    color_desc, hog_desc = get_descriptors(path)
    valid_concat_color_pyramid_desc.append(color_desc)
    valid_hog_desc.append(hog_desc)


assert len(train_labels) == train_images_num, 'La cantidad de etiquetas que tiene debe ser igual a la cantidad de imagenes que usted especificó'
assert len(train_concat_color_pyramid_desc[0]) > 1, 'El descriptor debe tener más de una dimensión'
assert len(train_hog_desc[0]) > 1, 'El descriptor debe tener más de una dimensión'
assert len(train_concat_color_pyramid_desc[0]) <= 7056, 'Su descriptor unimodal de color tiene demasiadas dimensiones'
assert len(train_hog_desc[0]) <= 7056, 'Su descriptor unimodal de forma tiene demasiadas dimensiones'
assert len(train_concat_color_pyramid_desc) == len(train_hog_desc), 'Usted debe tener la misma cantidad de descriptores en todas las listas'
assert len(np.unique(train_labels)) == 5, 'Solo deben haber 5 etiquetas'
assert len(np.unique(valid_labels)) == 5, 'Solo deben haber 5 etiquetas'

assert len(valid_labels) == 1723, 'Usted debe sacar el descriptor de todas las imagenes de validación'
assert len(valid_hog_desc[0]) == len(train_hog_desc[0]), 'El descriptor del histograma concatenado en train y valid debe ser igual'

print(f"Las dimensiones de su descriptor de color son {np.array(train_concat_color_pyramid_desc).shape[1]}")
print(f"Las dimensiones de su descriptor de forma son {np.array(train_hog_desc).shape[1]}")


svm_kernel1_C1_concatcolorPyram='' #SVM con un kernel 1 especificado con C1 especificado y entrenado con concat color en piramide
svm_kernel2_C1_concatcolorPyram='' #SVM con un kernel 2 especificado con C1 especificado y entrenado con concat color en piramide
svm_kernel1_C2_concatcolorPyram='' #SVM con un kernel 1 especificado con C2 especificado y entrenado con concat color en piramide
svm_kernel2_C2_concatcolorPyram='' #SVM con un kernel 2 especificado con C2 especificado y entrenado con concat color en piramide

svm_kernel1_C1_hog='' #SVM con un kernel 1 especificado con C1 especificado y entrenado con hog
svm_kernel2_C1_hog='' #SVM con un kernel 2 especificado con C1 especificado y entrenado con hog
svm_kernel1_C2_hog='' #SVM con un kernel 1 especificado con C2 especificado y entrenado con hog
svm_kernel2_C2_hog='' #SVM con un kernel 2 especificado con C2 especificado y entrenado con hog

# YOUR CODE HERE (Daniela)
kernel1, kernel2 = "linear", "rbf"   # Kernels a comparar
C1, C2 = 1, 10                       # Valores de C a comparar

# Modelos con el descriptor de color en pirámide
svm_kernel1_C1_concatcolorPyram = SVC(kernel=kernel1, C=C1).fit(train_concat_color_pyramid_desc, train_labels)
svm_kernel2_C1_concatcolorPyram = SVC(kernel=kernel2, C=C1).fit(train_concat_color_pyramid_desc, train_labels)
svm_kernel1_C2_concatcolorPyram = SVC(kernel=kernel1, C=C2).fit(train_concat_color_pyramid_desc, train_labels)
svm_kernel2_C2_concatcolorPyram = SVC(kernel=kernel2, C=C2).fit(train_concat_color_pyramid_desc, train_labels)

# Modelos con el descriptor HOG
svm_kernel1_C1_hog = SVC(kernel=kernel1, C=C1).fit(train_hog_desc, train_labels)
svm_kernel2_C1_hog = SVC(kernel=kernel2, C=C1).fit(train_hog_desc, train_labels)
svm_kernel1_C2_hog = SVC(kernel=kernel1, C=C2).fit(train_hog_desc, train_labels)
svm_kernel2_C2_hog = SVC(kernel=kernel2, C=C2).fit(train_hog_desc, train_labels)


assert not svm_kernel1_C1_concatcolorPyram.fit_status_, 'Su modelo no esta entrenado'
assert not svm_kernel2_C1_concatcolorPyram.fit_status_, 'Su modelo no esta entrenado'
assert not svm_kernel1_C2_concatcolorPyram.fit_status_, 'Su modelo no esta entrenado'
assert not svm_kernel2_C2_concatcolorPyram.fit_status_, 'Su modelo no esta entrenado'

assert not svm_kernel1_C1_hog.fit_status_, 'Su modelo no esta entrenado'
assert not svm_kernel2_C1_hog.fit_status_, 'Su modelo no esta entrenado'
assert not svm_kernel1_C2_hog.fit_status_, 'Su modelo no esta entrenado'
assert not svm_kernel2_C2_hog.fit_status_, 'Su modelo no esta entrenado'

#Caracteristicas
assert len(train_concat_color_pyramid_desc[0]) == svm_kernel1_C1_concatcolorPyram.n_features_in_, 'Su modelo no esta entrenado con el descriptor correcto'
assert len(train_concat_color_pyramid_desc[0]) == svm_kernel2_C1_concatcolorPyram.n_features_in_, 'Su modelo no esta entrenado con el descriptor correcto'
assert len(train_concat_color_pyramid_desc[0]) == svm_kernel1_C2_concatcolorPyram.n_features_in_, 'Su modelo no esta entrenado con el descriptor correcto'
assert len(train_concat_color_pyramid_desc[0]) == svm_kernel2_C2_concatcolorPyram.n_features_in_, 'Su modelo no esta entrenado con el descriptor correcto'

assert len(train_hog_desc[0]) == svm_kernel1_C1_hog.n_features_in_, 'Su modelo no esta entrenado con el descriptor correcto'
assert len(train_hog_desc[0]) == svm_kernel2_C1_hog.n_features_in_, 'Su modelo no esta entrenado con el descriptor correcto'
assert len(train_hog_desc[0]) == svm_kernel1_C2_hog.n_features_in_, 'Su modelo no esta entrenado con el descriptor correcto'
assert len(train_hog_desc[0]) == svm_kernel2_C2_hog.n_features_in_, 'Su modelo no esta entrenado con el descriptor correcto'


predict_svm_kernel1_C1_concatcolorPyram='' #Predict de modelo con con un kernel 1 especificado con C1 especificado y entrenado con concat color en piramide
predict_svm_kernel2_C1_concatcolorPyram='' #Predict de modelo con con un kernel 2 especificado con C1 especificado y entrenado con concat color en piramide
predict_svm_kernel1_C2_concatcolorPyram='' #Predict de modelo con con un kernel 1 especificado con C2 especificado y entrenado con concat color en piramide
predict_svm_kernel2_C2_concatcolorPyram='' #Predict de modelo con con un kernel 2 especificado con C2 especificado y entrenado con concat color en piramide

predict_svm_kernel1_C1_hog='' #Predict de modelo con con un kernel 1 especificado con C1 especificado y entrenado con hog
predict_svm_kernel2_C1_hog='' #Predict de modelo con con un kernel 2 especificado con C1 especificado y entrenado con hog
predict_svm_kernel1_C2_hog='' #Predict de modelo con con un kernel 1 especificado con C2 especificado y entrenado con hog
predict_svm_kernel2_C2_hog='' #Predict de modelo con con un kernel 2 especificado con C2 especificado y entrenado con hog

# YOUR CODE HERE (Daniela)
# Predicciones sobre validación con el mismo descriptor usado en el entrenamiento
predict_svm_kernel1_C1_concatcolorPyram = svm_kernel1_C1_concatcolorPyram.predict(valid_concat_color_pyramid_desc)
predict_svm_kernel2_C1_concatcolorPyram = svm_kernel2_C1_concatcolorPyram.predict(valid_concat_color_pyramid_desc)
predict_svm_kernel1_C2_concatcolorPyram = svm_kernel1_C2_concatcolorPyram.predict(valid_concat_color_pyramid_desc)
predict_svm_kernel2_C2_concatcolorPyram = svm_kernel2_C2_concatcolorPyram.predict(valid_concat_color_pyramid_desc)

predict_svm_kernel1_C1_hog = svm_kernel1_C1_hog.predict(valid_hog_desc)
predict_svm_kernel2_C1_hog = svm_kernel2_C1_hog.predict(valid_hog_desc)
predict_svm_kernel1_C2_hog = svm_kernel1_C2_hog.predict(valid_hog_desc)
predict_svm_kernel2_C2_hog = svm_kernel2_C2_hog.predict(valid_hog_desc)


assert len(predict_svm_kernel1_C1_concatcolorPyram)==1723, 'Deben tener 1723 predicciones en validación'
assert len(predict_svm_kernel1_C1_hog)==1723, 'Deben tener 1723 predicciones en validación'
assert len(np.unique(predict_svm_kernel1_C1_hog))==5, 'Deberían haber 5 etiquetas en sus predicciones'
assert np.mean(np.std([predict_svm_kernel1_C1_hog,predict_svm_kernel2_C1_hog,predict_svm_kernel1_C2_hog,predict_svm_kernel2_C2_hog],axis=1))>0.1, 'Sus predicciones son iguales'


f_kernel1_C1_concatcolorPyram='' #F medida de modelo kernel1 C1 y concatcolorPyram 
f_kernel2_C1_concatcolorPyram='' #F medida de modelo kernel2 C1 y concatcolorPyram 
f_kernel1_C2_concatcolorPyram='' #F medida de modelo kernel1 C2 y concatcolorPyram 
f_kernel2_C2_concatcolorPyram='' #F medida de modelo kernel2 C2 y concatcolorPyram 

f_kernel1_C1_hog='' #F medida de modelo kernel1 C1 y hog 
f_kernel2_C1_hog='' #F medida de modelo kernel2 C1 y hog 
f_kernel1_C2_hog='' #F medida de modelo kernel1 C2 y hog 
f_kernel2_C2_hog='' #F medida de modelo kernel2 C2 y hog 

# YOUR CODE HERE (Daniela)
def f1_macro(pred):
    """F1 promedio (macro) entre todas las clases de validación."""
    return precision_recall_fscore_support(valid_labels, pred, average="macro", zero_division=0)[2]

f_kernel1_C1_concatcolorPyram = f1_macro(predict_svm_kernel1_C1_concatcolorPyram)
f_kernel2_C1_concatcolorPyram = f1_macro(predict_svm_kernel2_C1_concatcolorPyram)
f_kernel1_C2_concatcolorPyram = f1_macro(predict_svm_kernel1_C2_concatcolorPyram)
f_kernel2_C2_concatcolorPyram = f1_macro(predict_svm_kernel2_C2_concatcolorPyram)

f_kernel1_C1_hog = f1_macro(predict_svm_kernel1_C1_hog)
f_kernel2_C1_hog = f1_macro(predict_svm_kernel2_C1_hog)
f_kernel1_C2_hog = f1_macro(predict_svm_kernel1_C2_hog)
f_kernel2_C2_hog = f1_macro(predict_svm_kernel2_C2_hog)

# Tabla resumen de los 8 experimentos
print("{:<12} | {:<8} {:<5} | {:<8}".format("Descriptor", "Kernel", "C", "F1"))
print("-" * 40)
for desc, k, c, f in [("Color pir.", kernel1, C1, f_kernel1_C1_concatcolorPyram),
                      ("Color pir.", kernel2, C1, f_kernel2_C1_concatcolorPyram),
                      ("Color pir.", kernel1, C2, f_kernel1_C2_concatcolorPyram),
                      ("Color pir.", kernel2, C2, f_kernel2_C2_concatcolorPyram),
                      ("HOG", kernel1, C1, f_kernel1_C1_hog),
                      ("HOG", kernel2, C1, f_kernel2_C1_hog),
                      ("HOG", kernel1, C2, f_kernel1_C2_hog),
                      ("HOG", kernel2, C2, f_kernel2_C2_hog)]:
    print("{:<12} | {:<8} {:<5} | {:<8.4f}".format(desc, k, c, f))


final_model='' #Modelo final entrenado
final_train_descriptor='' #Lista de descriptores en su modelo final en entrenamiento
final_valid_descriptor='' #Lista de descriptores en su modelo final en validación
# YOUR CODE HERE (Daniela)
# Mejor modelo: color en pirámide (HSV) con kernel RBF y C = 10 (F1 ≈ 0.60 en validación)
final_model = svm_kernel2_C2_concatcolorPyram
final_train_descriptor = train_concat_color_pyramid_desc
final_valid_descriptor = valid_concat_color_pyramid_desc


final_predict=final_model.predict(final_valid_descriptor)



_,_,f_final,_=precision_recall_fscore_support(valid_labels,final_predict,average='macro')

assert len(final_valid_descriptor[0])==final_model.n_features_in_, 'Su modelo no fue entrenado con el mismo descriptor con el que se evalua'
assert f_final>0.55, 'Necesita refinar su modelo'


train_multimodal_desc = "" #Descriptor multimodal de entrenamiento
val_multimodal_desc = "" #Descriptor multimodal de validación
svm_multimodal_desc = "" #Modelo SVM entrenado con descriptor multimodal
predict_svm_multimodal_desc = "" #Predicción de modelo SVM entrenado con descriptor multimodal
f_multimodal_desc = "" #F medida de modelo SVM entrenado con descriptor multimodal

# YOUR CODE HERE (Daniela)
# Se pegan, imagen por imagen, el descriptor de color en pirámide y el HOG
train_multimodal_desc = np.concatenate([np.array(train_concat_color_pyramid_desc), np.array(train_hog_desc)], axis=1)
val_multimodal_desc = np.concatenate([np.array(valid_concat_color_pyramid_desc), np.array(valid_hog_desc)], axis=1)

# SVM con los mejores hiperparámetros encontrados (kernel RBF, C = 10)
svm_multimodal_desc = SVC(kernel="rbf", C=10).fit(train_multimodal_desc, train_labels)

# Predicción y F1 macro en validación
predict_svm_multimodal_desc = svm_multimodal_desc.predict(val_multimodal_desc)
f_multimodal_desc = precision_recall_fscore_support(valid_labels, predict_svm_multimodal_desc, average="macro", zero_division=0)[2]

print(f"El valor de la F medida de su modelo multimodal es {f_multimodal_desc}")


assert len(train_multimodal_desc[0])==len(train_concat_color_pyramid_desc[0])+len(train_hog_desc[0]), 'Su descriptor multimodal no tiene la cantidad de dimensiones correctas'
assert svm_multimodal_desc.fit_status_==False, 'Su modelo no esta entrenado'
assert len(predict_svm_multimodal_desc)==1723, 'Deben tener 1723 predicciones en validación'
assert f_multimodal_desc>0.55, 'Necesita refinar su modelo multimodal'


from utils import *
# converter("Entrega 3")
