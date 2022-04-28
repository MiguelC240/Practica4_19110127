import numpy as np 
import cv2 
from matplotlib import pyplot as plt

import pixellib
from pixellib.instance import instance_segmentation


Img1 = cv2.imread('Imagen_Dia.jpg')
res1 = cv2.resize(Img1, dsize=(300, 300))


negro = np.zeros((300, 300, 3), dtype=np.uint8)
negro2 = np.zeros((300, 300, 3), dtype=np.uint8)
negro3 = np.zeros((300, 300, 3), dtype=np.uint8)



################ PACMAN ##################

#Poligono
pacman = np.array([[224,65],[59,65],[59,229],[224,229],[224,203],[175,179],[224,152]], np.int32)
cv2.polylines(negro, [pacman], True, (0,0,255), 5)

cv2.rectangle(negro,(175,83),(211,122),(0,0,255),5)#Rojo

cv2.imshow('Pacman', negro)


################ PERRO ##################


cabeza = np.array([[93,26],[40,89],[23,74],[78,8,],[93,26],[158,24],[160,25],[176,13],[222,80],[202,94],[160,28],[160,25],[158,24],[160,69],[172,69],[172,116],[123,116],[77,116],[77,67],[92,67]], np.int32)
cv2.polylines(negro2, [cabeza], True, (255,255,255), 3)

cuerpo = np.array([[124,68],[124,116],[164,278],[76,278],[124,116]], np.int32)
cv2.polylines(negro2, [cuerpo], True, (255,255,255), 3)

cv2.line(negro2,(92,68),(157,68),(255,255,255),3)#Blano

pata_der = np.array([[138,171],[221,229],[185,254],[215,292],[172,292],[185,254],[161,273],[138,171]], np.int32)
cv2.polylines(negro2, [pata_der], True, (255,255,255), 3)


pata_izq = np.array([[105,172],[23,221],[55,252],[25,286],[66,286],[55,252],[76,271],[105,172]], np.int32)
cv2.polylines(negro2, [pata_izq], True, (255,255,255), 3)

patas = np.array([[161,279],[161,289],[130,289],[129,279],[107,279],[107,289],[77,289],[77,279]], np.int32)
cv2.polylines(negro2, [patas], True, (255,255,255), 3)

cv2.circle(negro2,(105,59), 3, (255,255,255), -1)

cv2.circle(negro2,(144,59), 3, (255,255,255), -1)

cv2.imshow('Perro',negro2)




################ NOMBRE ##################

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(negro3,'Miguel Cortes',(30,80), font, 1, (132,140,58), 2, cv2.LINE_AA)
cv2.putText(negro3,'19110127',(50,160), font, 1, (132,140,58), 2, cv2.LINE_AA)


cv2.imshow('Nombre',negro3)



################ SEGMENTACIÓN DE IMÁGENES MEDIANTE OPERACIONES MORFOLÓGICAS ##################

gris = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original', gris) 

ret, thresh = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
cv2.imshow('Segmentacion', thresh) 


kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2) 
  
bg = cv2.dilate(closing, kernel, iterations = 1) 
  
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0) 
ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0) 
  
#cv2.imshow('Sin ruido', fg)
cv2.imwrite("Sin_ruido.jpg",fg)


################ ROI ##################


Sin_ruido = cv2.imread('Sin_ruido.jpg')


roi = cv2.selectROI(Sin_ruido)

print(roi)


Segmentada = Sin_ruido[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]


cv2.imshow("ROI", Segmentada)

cv2.imwrite("ROI.jpg",Segmentada)

cv2.waitKey(0)
cv2.destroyAllWindows()
