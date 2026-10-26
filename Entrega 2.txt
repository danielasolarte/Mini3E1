# Importar librerias importantes
import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt

from scipy.signal import correlate2d
from scipy.io import loadmat

import cv2

from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


group = None # Cambiar por el número de grupo
# YOUR CODE HERE (Daniela)
group = 19 # Número de grupo asignado en Bloque Neón (el mismo de la Entrega 1)


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


def color_hist(Im: np.ndarray, type_h: str, bins: int) -> np.ndarray:
    """Función que calcula el histograma de color conjunto o concatenado de una imagen

    Args:
        Im (numpy.ndarray): Imagen a la que calcular el histograma
        type_h (str): Tipo de histograma a calcular, puede ser "concat" o "joint"
        bins (int): Cantidad de bins que usar en el histograma. En el caso de concat el número de bins finale será 3*bins
        En el caso de Joint será bins^3

    Returns:
        numpy.ndarray: Histograma de color normalizado
    """
    h = "" # Histograma de color normalizado
    # YOUR CODE HERE (Daniela)
    # Como en el tutorial de la semana 11, el rango de los bins va del mínimo al máximo de cada canal
    if type_h == "concat":
        # Un histograma por cada canal y luego se ponen uno detrás del otro
        h = np.concatenate([np.histogram(Im[:, :, c], bins=bins)[0] for c in range(3)])
    elif type_h == "joint":
        # Un solo histograma 3D con todas las combinaciones de los 3 canales, luego se aplana
        h = np.histogramdd(Im.reshape(-1, 3), bins=bins)[0].flatten()

    # Se normaliza para que la suma total sea 1
    h = h / np.sum(h)
    return h


img=processing(cv2.imread(os.path.join(".", "dataset", "train", "Canis latrans", "f494fab00ebb2d87c8d3bf5215560d54.jpg")),  (224, 224), 'rgb')
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
assert np.isclose(np.std(concat_h),0.024,0.05), 'Su histograma de color concatenado es erroneo'
assert np.isclose(np.std(joint_h),0.0063,0.05), 'Su histograma de color conjunto es erroneo'


train_images_num='' #cantidad de imagens de entrenamiento
train_joint_color_desc=[] #Descriptor de imagenes con histogramas conjuntos de color
train_concat_color_desc=[] #Descriptor de imagenes con histogramas concatenados de color

valid_joint_color_desc=[] #Descriptor de imagenes con histogramas conjuntos de color
valid_concat_color_desc=[] #Descriptor de imagenes con histogramas concatenados de color

# YOUR CODE HERE (Daniela)
color_space = "hsv" # Espacio de color elegido
bins = 10                 # Bins por canal (concat: 3*bins, joint: bins^3)
img_size = (224, 224)          # Tamaño estándar de las imágenes

# Se usan todas las imágenes de entrenamiento
train_images_num = len(train_paths)

# Descriptores de entrenamiento (las etiquetas van en el mismo orden que las rutas)
for path in tqdm(train_paths[:train_images_num], desc="Train"):
    Im = processing(cv2.imread(path), img_size, color_space)
    train_joint_color_desc.append(color_hist(Im, "joint", bins))
    train_concat_color_desc.append(color_hist(Im, "concat", bins))
train_color_labels = train_labels[:train_images_num]

# Descriptores de validación, con el mismo procedimiento
for path in tqdm(valid_paths, desc="Valid"):
    Im = processing(cv2.imread(path), img_size, color_space)
    valid_joint_color_desc.append(color_hist(Im, "joint", bins))
    valid_concat_color_desc.append(color_hist(Im, "concat", bins))
valid_color_labels = valid_labels


assert len(train_joint_color_desc) == 5163, 'La cantidad de descriptores no es igual a la cantidad de imágenes totales'
assert len(train_concat_color_desc) == 5163, 'La cantidad de descriptores no es igual a la cantidad de imágenes totales'
assert len(valid_joint_color_desc) == 1723, 'La cantidad de descriptores no es igual a la cantidad de imágenes totales'
assert len(valid_concat_color_desc) == 1723, 'La cantidad de descriptores no es igual a la cantidad de imágenes totales'

assert len(train_joint_color_desc[0])<= 7056, 'Su descriptor unimodal de color tiene demasiadas dimensiones'
assert len(train_concat_color_desc[0])<= 7056, 'Su descriptor unimodal de color tiene demasiadas dimensiones'
assert len(train_joint_color_desc[0]) == len(train_joint_color_desc[0]), 'Los descriptores conjuntos tienen diferentes dimensiones'
assert len(train_concat_color_desc[0]) == len(valid_concat_color_desc[0]), 'Los descriptores concatenados tienen diferentes dimensiones'


NNClassifier_concatcolor='' # Clasificador con histogramas concatenados de color
NNClassifier_jointcolor='' # Clasificador con histogramas conjuntos de color

predict_NNClassifier_concatcolor = '' # Predicciones del clasificador con histogramas concatenados de color
predict_NNClassifier_jointcolor = '' # Predicciones del clasificador con histogramas conjuntos de color

# YOUR CODE HERE (Daniela)
k = 5 # Cantidad de vecinos

# 1. Se crea un clasificador K-NN para cada tipo de histograma
NNClassifier_concatcolor = KNeighborsClassifier(n_neighbors=k)
NNClassifier_jointcolor = KNeighborsClassifier(n_neighbors=k)

# 2. Se entrenan con los descriptores y etiquetas de entrenamiento
NNClassifier_concatcolor.fit(train_concat_color_desc, train_color_labels)
NNClassifier_jointcolor.fit(train_joint_color_desc, train_color_labels)

# 3. Se predicen las etiquetas de validación
predict_NNClassifier_concatcolor = NNClassifier_concatcolor.predict(valid_concat_color_desc)
predict_NNClassifier_jointcolor = NNClassifier_jointcolor.predict(valid_joint_color_desc)


assert NNClassifier_concatcolor.n_samples_fit_ == 5163, 'No se entrenó el clasificador con histogramas concatenados de color'
assert NNClassifier_concatcolor.n_features_in_ > 0, 'No se entrenó el clasificador con histogramas concatenados de color'
assert NNClassifier_jointcolor.n_samples_fit_ == 5163, 'No se entrenó el clasificador con histogramas conjuntos de color'
assert NNClassifier_jointcolor.n_features_in_ > 0, 'No se entrenó el clasificador con histogramas conjuntos de color'
assert NNClassifier_concatcolor.classes_.shape[0] == len(categories), 'El clasificador con histogramas concatenados de color no tiene las clases correctas'
assert NNClassifier_jointcolor.classes_.shape[0] == len(categories), 'El clasificador con histogramas conjuntos de color no tiene las clases correctas'

assert len(predict_NNClassifier_concatcolor) == len(valid_concat_color_desc), 'El número de predicciones debe ser igual al número de descriptores de validación'
assert len(predict_NNClassifier_jointcolor) == len(valid_joint_color_desc), 'El número de predicciones debe ser igual al número de descriptores de validación'


precision_concatcolor = '' #Precision del clasificador con histogramas concatenados de color
recall_concatcolor = '' #Recall del clasificador con histogramas concatenados de color
f1_score_concatcolor = '' #F1 del clasificador con histogramas concatenados de color

precision_jointcolor = '' #Precision del clasificador con histogramas conjuntos de color
recall_jointcolor = '' #Recall del clasificador con histogramas conjuntos de color
f1_score_jointcolor = '' #F1 del clasificador con histogramas conjuntos de color

# YOUR CODE HERE (Daniela)
# Métricas promedio (macro) comparando las etiquetas reales de validación con las predichas
precision_concatcolor, recall_concatcolor, f1_score_concatcolor, _ = precision_recall_fscore_support(
    valid_color_labels, predict_NNClassifier_concatcolor, average="macro", zero_division=0)
precision_jointcolor, recall_jointcolor, f1_score_jointcolor, _ = precision_recall_fscore_support(
    valid_color_labels, predict_NNClassifier_jointcolor, average="macro", zero_division=0)

print ("{:<25} | {:<14} {:<14} {:<14}".format('Modelo','Precisión','Cobertura', 'F1'))
print ('----------------------------------------------------------------------')
print ("{:<25} | {:<14.3f} {:<14.3f} {:<14.3f}".format('Histograma concatenado',precision_concatcolor,recall_concatcolor,f1_score_concatcolor))
print ("{:<25} | {:<14.3f} {:<14.3f} {:<14.3f}".format('Histograma Conjunto',precision_jointcolor,recall_jointcolor,f1_score_jointcolor))


assert f1_score_concatcolor > 0.28, 'El modelo con histogramas concatenados de color debería tener una f-medida mayor'
assert f1_score_jointcolor > 0.28, 'El modelo con histogramas cnojuntos de color debería tener una f-medida mayor'


Im_list = '' #lista de imagenes, 5 de cada categoría
# YOUR CODE HERE (Daniela)
random.seed(group) # Semilla para que siempre salgan las mismas imágenes
Im_list = []
for name, idx in dict_labels.items():
    # Rutas de entrenamiento de esta clase
    class_paths = [p for p, l in zip(train_paths, train_labels) if l == idx]
    # Se eligen 5 al azar, se redimensionan a 256x256 y se pasan a gris
    for path in random.sample(class_paths, 5):
        Im_list.append(processing(cv2.imread(path), (256, 256), "gray"))

print(f"Cantidad de imágenes en Im_list: {len(Im_list)}")


filterbank='' #arreglo de filtros
# YOUR CODE HERE (Daniela)
# loadmat devuelve un diccionario; la llave "filterbank" tiene la matriz 49x49x18
filterbank = loadmat("filterbank.mat")["filterbank"]
print(f"Tamaño del banco de filtros: {filterbank.shape}")


def filter_map(Im: np.ndarray, filterbank: np.ndarray) -> np.ndarray:
    """Funcion que aplica un banco de filtros a una imagen

    Args:
        Im (numpy.ndarray): Imagen de entrada RGB
        filterbank (numpy.ndarray): Banco de filtros

    Returns:
        numpy.ndarray: Resultado de aplicar el banco de filtros a la imagen
    """
    new_im = '' #Imagen con los filtros aplicados
    # YOUR CODE HERE (Daniela)
    # Si la imagen está a color (3 canales) se pasa a escala de grises
    if len(Im.shape) == 3:
        Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)

    # Se aplica cada filtro; "same" mantiene el tamaño de la imagen original
    responses = []
    for i in range(filterbank.shape[2]):
        responses.append(correlate2d(Im, filterbank[:, :, i], mode="same"))

    # Se juntan las N respuestas como los N canales de una sola imagen
    new_im = np.stack(responses, axis=2)
    return new_im


img=processing(cv2.imread(os.path.join(".", "dataset", "train", "Canis latrans", "f494fab00ebb2d87c8d3bf5215560d54.jpg")), (256, 256), 'gray')
new_img=filter_map(img,filterbank)

x,y=img.shape
x2,y2,f=new_img.shape
a,b,f2=filterbank.shape

assert f2==f, 'La imagen resultante debe tener la misma cantidad de canales como filtros en el filterbank'
assert x==x2 and y==y2, 'Las dimensiones de la imagen original no deben cambiar'
assert np.sum(np.std(new_img,axis=2))!=0, 'Usted usó el mismo filtro para todas las dimensiones'

assert np.sum(new_img[:,:,0]-correlate2d(img,filterbank[:,:,0],'same'))==0, 'El primer canal de new_im no es el resultado del filtrado con el primer filtro'
assert np.sum(new_img[:,:,2]-correlate2d(img,filterbank[:,:,2],'same'))==0, 'El primer canal de new_im no es el resultado del filtrado con el tercer filtro'

try:
    img_1=cv2.resize(cv2.imread(os.path.join(".", "dataset", "train", "Canis latrans", "f494fab00ebb2d87c8d3bf5215560d54.jpg")), (256, 256))
    filter_map(img_1,filterbank)
except:
    assert False, 'Su función debe funcionar con imágenes a color'

fig,ax=plt.subplots(2,3,figsize=(15,10))
ax[0,0].set_title('Imagen original')
ax[0,0].imshow(img, cmap='gray')
ax[0,1].set_title('Imagen filtrada 1')
ax[0,1].imshow(new_img[:,:,0])
ax[0,2].set_title('Imagen filtrada 10')
ax[0,2].imshow(new_img[:,:,9])

ax[0,0].axis('off')
ax[0,1].axis('off')
ax[0,2].axis('off')

ax[1,1].imshow(filterbank[:,:,0])
ax[1,1].set_title('Filtro 1')
ax[1,2].imshow(filterbank[:,:,9])
ax[1,2].set_title('Filtro 10')

ax[1,0].axis('off')
ax[1,1].axis('off')
ax[1,2].axis('off')

plt.show()


# YOUR CODE HERE (Daniela)
def pixel_dataset(Im_list: list[np.ndarray], filterbank: np.ndarray) -> np.ndarray:
    """Función que junta los píxeles de varias imágenes, cada uno con su respuesta a los filtros.

    Args:
        Im_list (list[np.ndarray]): Lista de imágenes (256x256)
        filterbank (numpy.ndarray): Banco de filtros

    Returns:
        numpy.ndarray: Arreglo de tamaño (cantidad de píxeles, N filtros)
    """
    Pixels = []
    for Im in tqdm(Im_list, desc="Pixeles"):
        # Respuesta de la imagen a los N filtros (256x256xN)
        fmap = filter_map(Im, filterbank)
        # Cada píxel pasa a ser un vector de N valores (65536xN)
        Pixels.append(fmap.reshape(-1, fmap.shape[2]))
    # Se ponen los píxeles de todas las imágenes uno debajo del otro
    return np.concatenate(Pixels, axis=0)


Pixels=pixel_dataset(Im_list,filterbank)

assert len(Pixels)==256*256*25, f'Deben tener una cantidad de 1638400 datos, pues son 256x256 pixeles x 25 imagenes, usted tiene {len(Pixels)}'
assert len(Pixels[0])==18, 'Las dimensiones de cada pixel deben ser 18, uno por cada filtro'
assert 0 in np.sum(np.array(Pixels)-filter_map(Im_list[0],filterbank)[0,0,:],axis=1), 'Debe agregar todos los pixeles de la imagen a su lista, puede que haya agregado mal los pixeles'


textones = 25 #cantidad de textones k
model1='' #modelo K means entrenado
# YOUR CODE HERE (Daniela)
# Modelo KMeans con 25 clústeres (textones); random_state para reproducibilidad
model1 = KMeans(n_clusters=textones, random_state=0)
# Se entrena con los píxeles de las 25 imágenes
model1.fit(Pixels)


assert len(model1.cluster_centers_) ==textones, 'Debe entrenar su modelo Kmeans con la cantidad de textones establecido'
assert model1.n_features_in_ == 18, 'Usted debe entrenar el modelo con el dataset de pixeles'


def texture_map(Im: np.ndarray, filterbank: np.ndarray, texton_dict: KMeans) -> np.ndarray:
    """Funcion que mapea una imagen a textones

    Args:
        Im (numpy.ndarray): Imagen de entrada RGB
        filterbank (numpy.ndarray): Banco de filtros
        texton_dict (KMeans): Diccionario de textones

    Returns:
        numpy.ndarray: Imagen mapeada a textones
    """
    texture_im = '' # Imagen mapeada a textones
    # YOUR CODE HERE (Daniela)
    # 1. Respuesta de la imagen a los filtros (alto x ancho x N)
    fmap = filter_map(Im, filterbank)
    h, w, n = fmap.shape

    # 2. Se pasa a una lista de píxeles (alto*ancho x N)
    pixels = fmap.reshape(-1, n)

    # 3. KMeans predice el textón de todos los píxeles al mismo tiempo
    labels = texton_dict.predict(pixels)

    # 4. Se vuelve a dar la forma de la imagen original
    texture_im = labels.reshape(h, w)
    return texture_im


img=processing(cv2.imread(os.path.join(".", "dataset", "train", "Canis latrans", "f494fab00ebb2d87c8d3bf5215560d54.jpg")),  (256, 256), 'gray')
texture_img=texture_map(img,filterbank,model1)

x,y=img.shape

assert len(texture_img.shape)==2, 'Su imagen debería tener solo 2 dimensiones'
x2,y2=texture_img.shape
assert x==x2 and y==y2, 'Las dimensiones de su imagen no deberían cambiar'
assert model1.predict([filter_map(img,filterbank)[0,0,:]])==texture_img[0,0],'La predicción del pixel 0,0 es erroneo'
assert model1.predict([filter_map(img,filterbank)[24,24,:]])==texture_img[24,24],'La predicción del pixel 24,24 es erroneo'

fig,ax=plt.subplots(1,2,figsize=(10,10))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[1].imshow(texture_img)

ax[0].axis('off')
ax[1].axis('off')
plt.show()


def texture_histogram(Im: np.ndarray, filterbank: np.ndarray, texton_dict: KMeans, bins: int) -> np.ndarray:
    """Funcion que calcula el histograma de textones de una imagen

    Args:
        Im (numpy.ndarray): Imagen de entrada RGB
        filterbank (numpy.ndarray): Banco de filtros
        texton_dict (KMeans): Diccionario de textones
        bins (int): Número de bins del histograma

    Returns:
        numpy.ndarray: Histograma de textones normalizado
    """
    histogram = "" # Histograma de textones normalizado
    # YOUR CODE HERE (Daniela)
    # Mapa de textones de la imagen
    texture_im = texture_map(Im, filterbank, texton_dict)

    # Se cuenta cuántas veces aparece cada textón, con rango (0, bins-1)
    histogram, _ = np.histogram(texture_im, bins=bins, range=(0, bins - 1))

    # Se normaliza para que la suma sea 1
    histogram = histogram / np.sum(histogram)
    return histogram


histogram_text=texture_histogram(img,filterbank,model1,textones)
assert np.isclose(np.sum(histogram_text),1), 'El histograma no esta normalizado'
assert len(histogram_text)==textones, 'El histograma no tiene la cantidad de bins que se indicó'


train_labels_list = [] #lista de etiquetas de entrenamiento
train_texture_desc=[] #vector de descriptores

valid_labels_list = [] #lista de etiquetas de validación
valid_texture_desc=[] #vector de descriptores
# YOUR CODE HERE (Daniela)
n_train_per_class = 30 # 30 imágenes x 5 clases = 150 de entrenamiento
n_valid_per_class = 10 # 10 imágenes x 5 clases = 50 de validación

random.seed(group) # Semilla para que siempre salgan las mismas imágenes
for name, idx in dict_labels.items():
    # Rutas de esta clase en train y en valid
    class_train = [p for p, l in zip(train_paths, train_labels) if l == idx]
    class_valid = [p for p, l in zip(valid_paths, valid_labels) if l == idx]

    # Descriptores de entrenamiento (imagen 256x256 en gris)
    for path in tqdm(random.sample(class_train, n_train_per_class), desc=f"Train {name}"):
        Im = processing(cv2.imread(path), (256, 256), "gray")
        train_texture_desc.append(texture_histogram(Im, filterbank, model1, textones))
        train_labels_list.append(idx)

    # Descriptores de validación
    for path in tqdm(random.sample(class_valid, n_valid_per_class), desc=f"Valid {name}"):
        Im = processing(cv2.imread(path), (256, 256), "gray")
        valid_texture_desc.append(texture_histogram(Im, filterbank, model1, textones))
        valid_labels_list.append(idx)

# Se guardan los descriptores y etiquetas con pickle para no tener que recalcularlos
with open("texture_descriptors.pkl", "wb") as f:
    pickle.dump({"train_desc": train_texture_desc, "train_labels": train_labels_list,
                 "valid_desc": valid_texture_desc, "valid_labels": valid_labels_list}, f)


assert len(valid_texture_desc[0]) == len(train_texture_desc[0]), 'Los descriptores no son de la misma dimensión'
assert len(valid_texture_desc) == len(valid_labels_list), 'Usted no cargó la cantidad correcta de etiquetas'
assert len(train_texture_desc) == len(train_labels_list), 'Usted no cargó la cantidad correcta de etiquetas'


predict_vector=[] #vector de etiquetas predichas
NNclassifier = '' #modelo de clasificador
# YOUR CODE HERE (Daniela)
# Clasificador K-NN entrenado con los histogramas de textones
NNclassifier = KNeighborsClassifier(n_neighbors=k)
NNclassifier.fit(train_texture_desc, train_labels_list)

# Predicción sobre los descriptores de validación
predict_vector = NNclassifier.predict(valid_texture_desc)


assert len(predict_vector)==len(valid_texture_desc), 'El vector de predicciones debe la misma cantidad que descriptores de validación'
assert NNclassifier.n_features_in_== 25, 'El modelo no tiene la cantidad de textones correcta'
assert NNclassifier.n_samples_fit_ == 150, 'El modelo no fue entrenado correctamente'


precision='' #Vector de precisión de cada categoría
recall='' #Vector de cobertura de cada categoría
f1_score='' #Vector de f medida de cada categoría
# YOUR CODE HERE (Daniela)
# Métricas por categoría (average=None) para el análisis
precision_cat, recall_cat, f1_cat, _ = precision_recall_fscore_support(
    valid_labels_list, predict_vector, average=None, zero_division=0)
for name, idx in dict_labels.items():
    print(f"{name:<28} P={precision_cat[idx]:.3f}  R={recall_cat[idx]:.3f}  F1={f1_cat[idx]:.3f}")

# Métricas promedio (macro), que son las que se comparan con las de color
precision, recall, f1_score, _ = precision_recall_fscore_support(
    valid_labels_list, predict_vector, average="macro", zero_division=0)
print(f"\nPromedio (macro): Precisión={precision:.3f}  Cobertura={recall:.3f}  F1={f1_score:.3f}")


assert f1_score > 0.16, 'La f-medida no es la esperada'


import torch

def filter_map(Im: np.ndarray, filterbank: np.ndarray) -> torch.tensor:
    """Funcion que aplica un banco de filtros a una imagen

    Args:
        Im (numpy.ndarray): Imagen de entrada RGB

    Returns:
        numpy.ndarray: Resultado de aplicar el banco de filtros a la imagen
    """
    if len(Im.shape) == 3:
        Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    Im = torch.tensor(Im, dtype=dtype, device=device)
    Im = Im[None, None]
    filterbank = torch.tensor(filterbank, dtype=dtype, device=device)
    filterbank = filterbank.permute(2, 0, 1)[:, None]
    convolution = torch.nn.functional.conv2d(Im, filterbank, padding="same")
    convolution = convolution.permute(0, 2, 3, 1)
    new_im = convolution.to("cpu").numpy()[0]

    return new_im


from utils import *
# converter("Entrega 2")
