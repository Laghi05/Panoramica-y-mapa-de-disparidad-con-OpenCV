import cv2 as cv

def ftMatching(imagenes_grises: list,  kp_des: dict, ref: int = 1):
    imgs_ftMatched = []
    Matches = {}
    
    nombres_imgs = list(kp_des.keys())
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    
    #Imagen de referencia
    nRef = nombres_imgs[ref]
    imgRef = imagenes_grises[ref]
    kpRef, desRef = kp_des.get(nRef, (None, None))
    
    #Iteracion sobre imagenes de prueba
    for i in range (len(imagenes_grises)):
        if i == ref:
            continue
        nTest = nombres_imgs[i]
        imgTest = imagenes_grises[i]
        kpTest, desTest = kp_des.get(nTest, (None, None))
    
    #Matching (referencia vs prueba)
        matches = bf.match(desRef, desTest)
        matches = sorted(matches, key=lambda x: x.distance)
        
        #Visualizacion
        num_matches_a_dibujar = min(50, len(matches))       
        img_matches = cv.drawMatches(
                imgRef, kpRef, 
                imgTest, kpTest, 
                matches[:num_matches_a_dibujar], 
                None, 
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        #Almacenar resultados
        clave_pareja = (nRef, nTest)
        Matches[clave_pareja] = matches
        
        imgs_ftMatched.append(img_matches)
    return imgs_ftMatched, Matches, nRef
 
def template_matching(imgs_grises: list, template: str, metodos: list = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED,
                                                      cv.TM_CCORR, cv.TM_CCORR_NORMED,
                                                      cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]):
    #Como distintos metodos pueden funcionar mejor o peor dependiendo del patron que se
    #quiera identificar, cuando no se especifica un metodo para el matching
    #la funcion prueba con todos los valores default disponibles y eso permite que uno mismo
    #pueda escoger el que mejor se adapta al patron.
    
    imgs_temp_matched = []
    img_template = cv.imread(template)
    temp_gris = cv.cvtColor(img_template, cv.COLOR_BGR2GRAY)
    h, w = temp_gris.shape
    
    for img in imgs_grises:
        for metodo in metodos:
            #Copia de la imagen
            img2 = img.copy()
            
            #Identifica la coincidencia
            resultado = cv.matchTemplate(img2, temp_gris, metodo)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(resultado)
            
            #Localiza el objeto
            if metodo in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                location = min_loc
            else:
                location = max_loc
            
            #Dibuja un rectangulo alrededor del objeto en la copia de la imagen
            esq_inf_derecha = (location[0] + w, location[1] + h)
            cv.rectangle(img2, location, esq_inf_derecha, 255, 5)
            
            imgs_temp_matched.append(img2)
            #Devuelve la lista de imagenes con el rectangulo dibujado
    return imgs_temp_matched