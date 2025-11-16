import cv2 as cv
import os
from pathlib import Path
from Modulo1 import cargar_imagenes, reducir_ruido, realzar_detalles
from Modulo2 import escala_grises, detectar_bordes, detectar_keypoints
from Modulo3 import ftMatching, template_matching
from Modulo4 import homografia, panoramica_manual, stitcher_pan

def mostrar_imagenes(titulo, imagenes:list):
    for i in imagenes:
        cv.imshow(titulo, i)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
img_template = "imagenes/panoramica/altavoz.png"
ruta_destino = Path(__file__).parent/'imagenes'/'panoramica'/'panoramica.jpg'
  

###### EJECUTE MAIN.PY CON CADA VERSION DE ESTOS BLOQUES EN EL ORDEN INDICADO PARA
#              EXPERIMENTAR LO QUE SE PUEDE LOGRAR CON ESTE CODIGO              

#PRIMERA EJECUCION
imagenes = ["imagenes/panoramica/1.jpg", "imagenes/panoramica/2.jpg"]    
ref = 0 #indice imagen de referencia
proy = 1 #indice imagen de proyeccion (homografia)

#SEGUNDA EJECUCION (comente la anterior)
#imagenes = ["/imagenes/panoramica/panoramica.jpg", "/imagenes/panoramica/3.jpg"]

#TERCERA EJECUCION (si usa tres o mas imagenes a la vez, la panoramica final sera generada por un algoritmo)
#imagenes = ["/imagenes/panoramica/1.jpg", "/imagenes/panoramica/2.jpg", "/imagenes/panoramica/3.jpg"]
#ref = 1 #descomente esto tambien

def crear_panoramica(imagenes: list):
    #Cargar
    imgs = cargar_imagenes(imagenes)
    mostrar_imagenes('Imagenes originales', imgs)
        
    #Acondicionamiento
    imgs_sin_ruido = reducir_ruido(imgs, 1, 1)
    mostrar_imagenes('Reduccion de ruido', imgs_sin_ruido)
        
    imgs_detalles_realzados = realzar_detalles(imgs, imgs_sin_ruido)
    mostrar_imagenes('Detalles realzados', imgs_detalles_realzados)
        
    imgs_grises = escala_grises(imgs_detalles_realzados)
    mostrar_imagenes('Escala de grises', imgs_grises)
        
    #Deteccion de bordes
    imgs_grises_suave = escala_grises(imgs_sin_ruido)
    imgs_bordes = detectar_bordes(imgs_grises_suave, 35, 50)
    mostrar_imagenes('Deteccion de bordes', imgs_bordes)
        
    #Deteccion de keypoints
    imgs_kp, kp_des = detectar_keypoints(imgs_grises)
    mostrar_imagenes('Keypoints detectados', imgs_kp)
        
    #Feature matching
    imgs_ftMatched, Matches, nRef  = ftMatching(imgs_grises, kp_des, ref)
    mostrar_imagenes('Feature Matching', imgs_ftMatched)
        
    #Template matching
    imgs_temp_matched = template_matching(imgs_grises, img_template, [cv.TM_CCOEFF_NORMED])
    mostrar_imagenes('Template Matching', imgs_temp_matched)
    
    #Calculo de matrices de homografia
    homografias = homografia(Matches, kp_des, nRef)
    print('Matriz de homografia:')
    print(homografias)
        
    if len(imagenes) == 2:
        
        #Generacion de panoramica usando matrices de homografia
        panoramica = panoramica_manual(imgs, homografias, ref, proy)
        
    elif len(imagenes) >= 3:
        #Cargar imagenes
        imgs = cargar_imagenes(imagenes)
        
        #Generar panoramica con algoritmo Stitcher
        panoramica = stitcher_pan(imgs)
    else:
        print('Por lo menos dos fotos')
        
    cv.imwrite(str(ruta_destino), panoramica)
    
crear_panoramica(imagenes)
os.startfile(str(ruta_destino))