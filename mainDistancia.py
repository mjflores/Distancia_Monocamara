import cv2
from ultralytics import YOLO
#from distanciaCamara import estimate_depth
from distanciaCamara import DistanciaMonoCamara
import numpy as np
import time

def detect_persons(dir_video):
    # Cargar el modelo YOLOv8
    model = YOLO('yolov8n.pt')  # Usamos el modelo nano por ser ligero
    distanciaMonoCamara = DistanciaMonoCamara(model_type="MiDaS_small", resize=(256, 256), depth_range=(1.0, 100.0), use_cuda=True)

    # Inicializar la cámara
    cap = cv2.VideoCapture(dir_video)
    
    while True:
        # Leer frame de la cámara
        ret, frame = cap.read()
        if not ret:
            break
            
        # Realizar la detección
        results = model.predict(frame, classes=[0])  # Solo detectar peatones (clase 0 en COCO)
        
        # Obtener las detecciones
        detections = results[0].boxes.data
        
        # Crear una copia del frame para mostrar las ROIs
        annotated_frame = frame.copy()        
        # Procesar cada detección
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            
            # Extraer la ROI
            roi = frame[y1:y2, x1:x2]
            t1 = time.time()
            distancia = distanciaMonoCamara.get_distance_from_roi(roi)
            t2 = time.time()
            print(f"Tiempo de distancia: {t2 - t1:.2f} segundos")
            # Calcular la profundidad usando la función de distanciaCamara
            #depth_map = estimate_depth(roi)
            
            # Mostrar la profundidad en el frame principal
            depth_text = f'Dist: {distancia:.2f}'
            cv2.putText(annotated_frame, depth_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Dibujar el bounding box en el frame principal
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
            #print(f"distancia ={distancia} cm,  depth_map = { depth_map} mts")
        # Mostrar el frame con los bounding boxes
        cv2.imshow('Detector de Personas', annotated_frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Iniciando detector de personas con YOLOv8...")
    dir_video = "C:\\Users\\mjflores\\MEGA\\Mis Programas\\Datos\\VideosConductor\\conductor1.avi"
    detect_persons(dir_video)
