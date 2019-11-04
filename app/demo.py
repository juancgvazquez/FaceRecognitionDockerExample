import os
from keras.preprocessing import image as kerasimg
from keras.models import model_from_json
import cv2
import face_recognition
import re
import uuid
import datetime
import pickle
import numpy as np
import logging
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)


# Demostración de Reconocimiento de video desde la camara web.
# Funciona con dos ajustes para hacerlo más dinámico:
# 1. Procesa cada cuadro a 1/4 de la resolución
# 2. Sólo detecta los rostros en un cuadro de por medio

# Selecciona la cámara a caṕturar
video_capture = cv2.VideoCapture(0)
# Crea listas con los nombre y los encodings de rostros conocidos
known_face_encodings = []
known_face_names = []

# Inicializar algunas variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Calculo los encodings por cara
for person in os.listdir('faces'):
    print("loading {}".format("faces/" + person))
    image = face_recognition.load_image_file("faces/" + person)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_names.append(re.sub('.jpg|.jpeg|.png', '', person))
    known_face_encodings.append(encoding)

model = model_from_json(
    open("facialexpression/facial_expression_model_structure.json", "r").read())
model.load_weights(
    'facialexpression/facial_expression_model_weights.h5')  # load weights


class Metric():
    def __init__(self, name, time):
        self.name = name
        self.emotions = [0, 0, 0, 0, 0, 0, 0]
        self.descriptions = ['enojado', 'disgustado', 'miedo',
                             'feliz', 'triste', 'sorprendido', 'neutral']
        self.arrival = time
        self.latest = None

    def update(self, time, emotion_index):
        self.emotions[emotion_index] += 1
        self.latest = time

    def print_info(self):
        print(self.name)
        print("emotion, count")
        for each in zip(self.descriptions, self.emotions):
            print("{}: {}".format(each[0], each[1]))


metricsDict = {}

while True:
    # Captura un cuadro de video
    ret, frame = video_capture.read()

    # Achica el cuadro para hacer un procesamiento más rápido
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_frame = frame
    # Convierte de BGR a RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    # Procesa cuadro de por medio
    if process_this_frame:
        # Ubica tdos los rotrostros y sus encodings en el frame actual
        face_locations = face_recognition.face_locations(
            rgb_small_frame, number_of_times_to_upsample=1, model='linear')
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # Ve si la cara matchea con alguna conocida
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)

            # Si se encuentra un match, usa el primero que encuentra
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                name = str(uuid.uuid4())
                known_face_names.append(name)
                known_face_encodings.append(face_encoding)

            face_names.append(name)

            if name not in metricsDict.keys():
                metricsDict[name] = Metric(name, datetime.datetime.now())

    # Para saltear frames
    process_this_frame = not process_this_frame

    # Muestra los resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Escala de nuevo las ubicaciones del rostro, porque estaba en 1/2
        # top *= 2
        # right *= 2
        # bottom *= 2
        # left *= 2

        # corta la región del rostro
        detected_face = frame[top:bottom, left:right]
        detected_face = cv2.cvtColor(
            detected_face, cv2.COLOR_BGR2GRAY)  # la transforma a gris
        # escala el tamaño a 48 por 48
        detected_face = cv2.resize(detected_face, (48, 48))
        img_pixels = kerasimg.img_to_array(detected_face)

        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)

        # encuentra el máximo indice
        max_index = np.argmax(predictions[0])
        emotions = ('enojado', 'disgustado', 'miedo',
                    'feliz', 'triste', 'sorprendido', 'neutral')
        emotion = emotions[max_index]
        text = "{}: {}%".format(
            emotion, int(predictions[0][max_index]*100))

        # Dibuja una caja alrededor del rostro
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        # Pone las etiquetas con un nombre bajo la cara
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom-20),
                    font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, text, (left + 6, bottom-2),
                    font, 0.5, (255, 255, 255), 1)

        metricsDict[name].update(datetime.datetime.now(), max_index)
        # metricsDict[name].print_info()

    # Muestra la imagen resultante
    cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video',(1280,720))
    cv2.imshow('Video', frame)

    # Aprieta 'ESC' para salir
    if cv2.waitKey(1) == 27:
        break

for name in metricsDict.keys():
    metricsDict[name].print_info()

# Liberar la Cámara
video_capture.release()
cv2.destroyAllWindows()
