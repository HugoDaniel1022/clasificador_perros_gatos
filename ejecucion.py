from keras import models
import numpy as np
import cv2
import os
from app import labels

model = models.load_model('nombre de modelo credo .keras')

ruta_predict = 'ruta de la carpeta con fotos aleatorias a predecir'

width = 300
height = 300


for i in os.listdir(ruta_predict):
    my_image = cv2.imread(f'{ruta_predict}{i}')
    my_image_r = cv2.resize(my_image, (width, height))

    result = model.predict(np.array([my_image_r]))[0]

    porcentaje = max(result)*100

    grupo = labels[result.argmax()]

    print(grupo, round(porcentaje), i)
