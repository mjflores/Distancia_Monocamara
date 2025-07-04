# Distancia_Monocamara

Este repositorio presenta una implementación robusta y modular para la estimación de distancia objeto-cámara utilizando visión monocular. Se basa en MiDaS (Monocular Depth Estimation) , un modelo de deep learning, para obtener medidas precisas de profundidad en tiempo real, enfocado especialmente en personas.

🔍 Características principales:
Pipeline completo : desde la captura de video hasta la calibración geométrica y optimización de métricas de distancia.

Detección de objetos : se utiliza YOLOv8 para detectar personas en la escena y calcular la distancia que las separa de la cámara.

Código modular : fácilmente extensible a cámaras RGB-D o integrable con sistemas como SLAM o ROS .

Casos de uso : navegación autónoma, realidad aumentada, análisis espacial en entornos no estructurados y más.

Arquitectura secuencial : ideal para principiantes o integración progresiva en proyectos complejos.

🛠️ Tecnologías utilizadas:

MiDaS (Depth Estimation)

YOLOv8 (Object Detection)

Python

OpenCV

📌 Notas:

Parte del código ha sido generado o asistido por inteligencia artificial, y está pensado para fines educativos y de prototipo funcional.
