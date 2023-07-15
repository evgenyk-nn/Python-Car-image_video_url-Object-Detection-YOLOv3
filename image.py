"""
Этот код выполняет распознавание и классификацию объектов (автомобилей) на изображении с помощью модели YOLO.
Библиотека OpenCV используется для обработки изображений и модель YOLO для обнаружения объектов.
"""

import cv2
import numpy as np


# Загрузка модели YOLO
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights') # вместо yolov3-tiny.weights  подгрузить yolov3.weights !

# Загрузка классов объектов
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Задание парковочной зоны
parking_area = [[100, 100], [500, 100], [500, 400], [100, 400]]
#parking_area = [[20, 730], [450, 920], [480, 825], [1575, 980]]

# Задание дорожной зоны
road_area = [[50, 50], [550, 50], [550, 450], [50, 450]]
#road_area = [[300, 730], [450, 760], [1520, 825], [1575, 850]]

def apply_yolo_object_detection(img):
    """
    Применение YOLO для распознавания объектов на изображении
    :param img: исходное изображение
    :return: изображение с распознанными объектами и информацией о количестве автомобилей и свободных мест на парковке
    """
    # Преобразование изображения в blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)

    # Запуск прямого прохода через нейронную сеть
    net.setInput(blob)
    outputs = net.forward(get_output_layers(net))

    # Обнаружение и классификация объектов
    objects = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'car':
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                width = int(detection[2] * img.shape[1])
                height = int(detection[3] * img.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                objects.append({'class': 'car', 'coordinates': (x, y, width, height)})

    # Разметка парковочной зоны
    img = draw_parking_area(img)

    # Разметка дорожной зоны
    img = draw_road_area(img)

    # Распознавание объектов и их разметка
    for object_info in objects:
        img = draw_object(object_info, img)

    # Вывод информации о количестве объектов на парковке и количестве свободных мест
    num_vehicles = len(objects)
    num_free_spaces = len(parking_area) - num_vehicles

    cv2.putText(img, f"Vehicles: {num_vehicles}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(img, f"Free spaces: {num_free_spaces}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

def get_output_layers(net):
    """
    Получение имен выходных слоев YOLO
    :param net: нейронная сеть YOLO
    :return: имена выходных слоев
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_object(object_info, img):
    """
    Разметка объекта на изображении
    :param object_info: информация о объекте
    :param img: исходное изображение
    :return: изображение с размеченным объектом
    """
    x, y, width, height = object_info['coordinates']
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(img, object_info['class'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

def is_selected_object_in_parking_area(object_info):
    """
    Проверка, находится ли выбранный объект в парковочной зоне
    :param object_info: информация о выбранном объекте
    :return: True, если объект находится в парковочной зоне, иначе False
    """
    x, y, width, height = object_info['coordinates']
    object_center_x = x + width / 2
    object_center_y = y + height / 2
    return cv2.pointPolygonTest(np.array(parking_area), (object_center_x, object_center_y), False) >= 0

def draw_parking_area(img):
    """
    Разметка парковочной зоны на изображении
    :param img: исходное изображение
    :return: изображение с размеченной парковочной зоной
    """
    pts = np.array(parking_area, np.int32)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    return img

def draw_road_area(img):
    """
    Разметка дорожной зоны на изображении
    :param img: исходное изображение
    :return: изображение с размеченной дорожной зоной
    """
    pts = np.array(road_area, np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 255), 2)
    return img

# Загрузка изображения
image = cv2.imread('parking3.png')


# Применение YOLO для распознавания объектов и разметка парковочной зоны
output_image = apply_yolo_object_detection(image)

# Вывод результата
cv2.imshow('Object Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()