import torch
import cv2
import numpy as np
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas = midas.to(device)
midas.eval()

# Transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((256, 256))  # Ajusta según modelo
    transforms.Resize((128, 128))  # Ajusta según modelo
])
# Función de inferencia de profundidad
def estimate_depth(frame):
    #img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame).unsqueeze(0)

    # Inferencia
    with torch.no_grad():
        depth_map = midas(input_tensor)

    # Normalizar mapa de profundidad
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #print("depth_map", depth_map.shape)
    avg_depth = np.mean(depth_map)
    return avg_depth

#========================================

class DistanciaMonoCamara:
    def __init__(self, model_type="MiDaS_small", resize=(128, 128), depth_range=(1.0, 100.0), use_cuda=True):
        """
        Inicializa el estimador de distancia con MiDaS.

        :param model_type: Modelo de MiDaS a usar ('MiDaS_small', 'DPT_Large', etc.)
        :param resize: Tamaño al que se redimensiona la imagen antes de la inferencia
        :param depth_range: Rango de distancia en metros (min, max)
        :param use_cuda: Usa GPU si es posible
        """
        self.resize = resize
        self.min_distance, self.max_distance = depth_range

        # Seleccionar dispositivo
        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

        # Cargar modelo
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas = self.midas.to(self.device)
        self.midas.eval()

        # Transformaciones
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize)
        ])

    def estimate_depth_map(self, frame):
        """
        Estima el mapa de profundidad de una imagen o ROI.

        :param frame: Imagen o ROI en formato numpy array (BGR OpenCV)
        :return: Mapa de profundidad normalizado (numpy array)
        """
        # frame es BGR
        # Redimensionar imagen
        img_resized = cv2.resize(frame, self.resize)
        #print("Size img_resized", img_resized.shape)
        #cv2.imshow("img_resized", img_resized)
        # Aplicar transformaciones y mover a dispositivo
        input_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        # Inferencia
        with torch.no_grad():
            depth_map = self.midas(input_tensor)

        # Procesar salida
        depth_map = depth_map.squeeze().cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return depth_map

    def get_distance_from_roi(self, roi_frame):
        """
        Devuelve la distancia estimada promedio de una región de interés (ROI).

        :param roi_frame: Región de la imagen donde hay un objeto (por ejemplo, un peatón)
        :return: Distancia estimada en metros
        """
        depth_map = self.estimate_depth_map(roi_frame)

        # Calcular profundidad promedio
        avg_depth_value = np.mean(depth_map)

        # Mapear profundidad normalizada a distancia real (método simple)
        distance = self.min_distance + (1.0 - avg_depth_value) * (self.max_distance - self.min_distance)

        return distance