import cv2 as cv
import numpy as np

def cargar_imagenes(rutas: list) -> list:
    imagenes = []
    
    for ruta in rutas:
        imagen = cv.imread(ruta)
        if imagen is None:
            print('No se cargo la imagen')
        else:
            imagenes.append(imagen)
    return imagenes

def reducir_ruido(imagenes: list, kernel_h = 9, kernel_w = 9) -> list:
    imagenes_sin_ruido = []
    
    for imagen in imagenes:
        img_prom = cv.blur(src=imagen, ksize=(kernel_h, kernel_w)) #filtro promedio
        img_gauss = cv.GaussianBlur(src=img_prom, ksize=(kernel_h+4, kernel_w+4), sigmaX=0) #filtro Gauss
        imagenes_sin_ruido.append(img_gauss)     
    return imagenes_sin_ruido

def realzar_detalles(imagenes: list, imagenes_sin_ruido: list, alpha: float = 2.0):
    imgs_detalles_realzados = []
    
    for original, suavizada in zip(imagenes, imagenes_sin_ruido):
        org_float = original.astype('float64')
        suav_float = suavizada.astype('float64')
              
        mascara_detalle = org_float - suav_float #Este es el realce en si
        #Usa la imagen suavizada para realzar los detalles de la original
        img_realzada_float = suav_float + alpha * mascara_detalle
        
        img_realzada_float = np.round(img_realzada_float)
        img_realzada_float = np.clip(img_realzada_float, 0, 255)
        
        img_final = img_realzada_float.astype('uint8')
        
        imgs_detalles_realzados.append(img_final)
    return imgs_detalles_realzados