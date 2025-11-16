import cv2 as cv

def escala_grises(imgs_acondicionadas: list) -> list:
    imgs_grises = []
    
    for imagen in imgs_acondicionadas:
        img_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
        imgs_grises.append(img_gris)
    return imgs_grises
                                    #Umbrales por defecto modificables al llamar a la funcion
def detectar_bordes(imgs_grises: list, threshold1: int = 150, threshold2: int = 200) -> list:
    imgs_bordedas = []
    
    for imagen in imgs_grises:
        img_bordes = cv.Canny(imagen, threshold1, threshold2) #Metodo Canny para deteccion de bordes
        imgs_bordedas.append(img_bordes)
    return imgs_bordedas

def detectar_keypoints(imgs_grises: list):
    imgs_kp = []
    kp_des = {}
    
    orb = cv.ORB_create()
    
    for idx, img in enumerate(imgs_grises):
        kp, des = orb.detectAndCompute(img, None)
        kp_img = cv.drawKeypoints(image=img, keypoints=kp, outImage=None, color=(0,255,0))
        
        imgs_kp.append(kp_img)
        
        #Como la funcion esta preparada para actuar sobre dos imagenes o mas (es escalable),
        #Se crea un diccionario que contiene la clave con los nombres asignados
        #de las imagenes, cuyo valor es una tupla con sus keypoints y descriptores
        nombre_temporal = f'img_{idx}'
        kp_des[nombre_temporal] = (kp, des)
    return imgs_kp, kp_des