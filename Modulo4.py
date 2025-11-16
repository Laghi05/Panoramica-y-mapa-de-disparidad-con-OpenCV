import cv2 as cv
import numpy as np

def homografia(Matches: dict, kp_des: dict, nRef: str):
    homografias = {}
    
    #Obtener puntos clave de la imagen de referencia
    kpRef, _ = kp_des.get(nRef, (None, None))
    
    #Iteracion sobre los resultados de las coincidencias (matches)
    for (n_ref, n_test), matches in Matches.items():
        if n_ref != nRef:
            continue
        #Obtener puntos clave de las imagenes de prueba
        kp_test, _ = kp_des.get(n_test, (None, None))
        
        #Extraer coordenadas de puntos coincidentes
        ptsRef = np.float32([kpRef[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        ptsTest = np.float32([kp_test[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        #Calcular homografia
        H, mask = cv.findHomography(ptsTest, ptsRef, cv.RANSAC, 3.0)
        
        #Almacenar resultados
        if H is not None and np.sum(mask) > 5:
            homografias[n_test] = H   
    return homografias

def panoramica_manual(imgs_grises: list, homografia_dict: dict, ref: int = 1, proy: int = 0):
    H = list(homografia_dict.values())[0]
    #Se toma la primera homografia del dicccionario a confianza porque este metodo solo se usara
    #cuando haya dos imagenes
    imgRef = imgs_grises[ref]
    imgProy = imgs_grises[proy]
    
    h_proy, w_proy = imgProy.shape[:2]
    h_ref, w_ref = imgRef.shape[:2]
    
    pts = np.float32([[0, 0], [0, h_proy], [w_proy, h_proy], [w_proy, 0]]).reshape(-1, 1, 2)
    pts_trans = cv.perspectiveTransform(pts, H)
    
    [x_min, y_min] = np.int32(pts_trans.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(pts_trans.max(axis=0).ravel() + 0.5)
    
    x_offset = min(0, x_min)
    y_offset = min(0, y_min)
    
    w_out = max(w_ref, x_max) - x_offset
    h_out = max(h_ref, y_max) - y_offset
    
    T = np.array([[1, 0, -x_offset], 
                  [0, 1, -y_offset], 
                  [0, 0, 1]])
    H_final = T @ H
    
    y_start = -y_offset
    y_end = -y_offset + h_ref
    x_start = -x_offset
    x_end = -x_offset + w_ref
     
    resultado_panoramica = cv.warpPerspective(imgProy, H_final, (w_out, h_out))
    
    mask = resultado_panoramica[y_start:y_end, x_start:x_end] == 0
    resultado_panoramica[y_start:y_end, x_start:x_end][mask] = imgRef[mask]
    
    return resultado_panoramica

def stitcher_pan(imagenes):
    stitcher = cv.Stitcher_create()
    (status, panorama_final) = stitcher.stitch(imagenes)
   
    if status == cv.Stitcher_OK:
        panoramica = panorama_final
    else:
        print('Error en Stitching')
    return panoramica

