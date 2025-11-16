import cv2 as cv
import numpy as np
from pathlib import Path

minDisparity = 0 #Desplazamiento minimo de pixeles para buscar coincidencias
numDisparities = 128 #valores de disparidad a buscar. Define el rango de disparidad
blockSize = 5 #tamaño del bloque de agregacion. Un valor mas alto hace la coincidencia mas robusta y el mapa mas suave
P1 = 200 #Controla la suavidad del mapa en areas planas
P2 = 800 #preserva los bordes de los objetos

stereo = cv.StereoSGBM_create(
    minDisparity = minDisparity,
    numDisparities = numDisparities,
    blockSize = blockSize,
    P1 = P1,
    P2 = P2
)

#Cargar imagenes estereo
imgL_color = cv.imread('imagenes/estereo/estereoL.jpg')
imgR_color = cv.imread('imagenes/estereo/estereoR.jpg')

#Convertir a escala de grises
imgL = cv.cvtColor(imgL_color, cv.COLOR_BGR2GRAY)
imgR = cv.cvtColor(imgR_color, cv.COLOR_BGR2GRAY)

#Ejecucion del algoritmo
disparity_mapa = stereo.compute(imgL, imgR)

#Escalamiento para obtener el valor de la disparidad en unidades de pixeles
disparity = disparity_mapa.astype(np.float32) / 16.0

#Normalizacion para poder visualizar la imagen correctamente
disparity_norm = ( disparity - minDisparity) / numDisparities

cv.imshow('leftview', imgL)
cv.imshow('righttview', imgR)
cv.imshow('Disparity', disparity_norm)

cv.waitKey(0)
cv.destroyAllWindows()