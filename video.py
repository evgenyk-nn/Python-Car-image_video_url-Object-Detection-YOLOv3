"""
Данный код используется для обнаружения и классификации объектов (автомобилей) на видео с помощью модели YOLO. Он загружает модель YOLO из файлов
'yolov3.cfg' и 'yolov3.weights', а также загружает список классов объектов из файла 'coco.names'. Затем он применяет модель YOLO к каждому кадру
видео, обнаруживает автомобили, размечает их на кадре и подсчитывает количество автомобилей и свободных мест на парковке. Код также определяет
парковочную зону и дорожную зону, размечает их на кадре видео и выводит информацию о количестве объектов на парковке и количестве свободных мест.
"""

import cv2
import numpy as np


# Загрузка модели YOLO
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')  # вместо yolov3-tiny.weights  подгрузить yolov3.weights !

# Загрузка классов объектов
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Задание парковочной зоны
#parking_area = [[100, 100], [500, 100], [500, 400], [100, 400]]
#parking_area = [[10, 150], [517, 140], [546, 322], [10, 350], [775, 153], [1686, 283], [1753, 380], [812, 299]]
#parking_area = [[10, 150], [517, 150], [546, 322], [10, 322], [775, 153], [1686, 283], [1753, 380], [812, 380]]
parking_area = [[10, 150], [1650, 200], [1900, 626], [10, 1000]]


# Задание дорожной зоны
#road_area = [[50, 50], [550, 50], [550, 450], [50, 450]]
#road_area = [[50, 400], [550, 400], [550, 750], [50, 750]]
road_area = [[20, 398], [1837, 384], [1864, 460], [25, 613]]

def apply_yolo_object_detection(frame):
    """
    Применение YOLO для распознавания объектов на кадре видео
    :param frame: исходный кадр видео
    :return: кадр с распознанными объектами и информацией о количестве автомобилей и свободных мест на парковке
    """
    # Преобразование кадра в blob
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)

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
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                objects.append({'class': 'car', 'coordinates': (x, y, width, height), 'confidence': confidence})

    # Применение алгоритма Non-Maximal Suppression
    objects = apply_non_maximal_suppression(objects)

    # Разметка парковочной зоны
    frame = draw_parking_area(frame)

    # Разметка дорожной зоны
    frame = draw_road_area(frame)

    # Распознавание объектов и их разметка
    for object_info in objects:
        frame = draw_object(object_info, frame)

    # Вывод информации о количестве объектов на парковке и количестве свободных мест
    num_vehicles = len(objects)
    num_free_spaces = len(parking_area) - num_vehicles

    cv2.putText(frame, f"Vehicles: {num_vehicles}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Free spaces: {num_free_spaces}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

def get_output_layers(net):
    """
    Получение имен выходных слоев YOLO
    :param net: нейронная сеть YOLO
    :return: имена выходных слоев
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def apply_non_maximal_suppression(objects, overlap_threshold=0.5):
    """
    Применение алгоритма Non-Maximal Suppression для удаления дублирующихся объектов
    :param objects: список объектов
    :param overlap_threshold: пороговое значение перекрытия для удаления объектов
    :return: отфильтрованный список объектов
    """
    if len(objects) == 0:
        return []

    # Сортировка объектов по убыванию уверенности (confidence score)
    objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)

    # Инициализация списка выбранных объектов
    selected_objects = []
    selected_objects.append(objects[0])

    # Расчет IoU (Intersection over Union) для всех объектов
    for i in range(1, len(objects)):
        box = objects[i]['coordinates']
        x, y, w, h = box
        area = w * h

        # Расчет IoU для текущего объекта с выбранными объектами
        overlap = []
        for selected_obj in selected_objects:
            selected_box = selected_obj['coordinates']
            selected_x, selected_y, selected_w, selected_h = selected_box
            selected_area = selected_w * selected_h

            # Расчет перекрытия (intersection)
            intersection_x = max(x, selected_x)
            intersection_y = max(y, selected_y)
            intersection_w = min(x + w, selected_x + selected_w) - intersection_x
            intersection_h = min(y + h, selected_y + selected_h) - intersection_y

            # Расчет IoU
            intersection_area = max(intersection_w, 0) * max(intersection_h, 0)
            iou = intersection_area / (area + selected_area - intersection_area)

            overlap.append(iou)

        # Если нет перекрытий с выбранными объектами, добавляем текущий объект в список выбранных
        if max(overlap) < overlap_threshold:
            selected_objects.append(objects[i])

    return selected_objects

def draw_object(object_info, frame):
    """
    Разметка объекта на кадре видео
    :param object_info: информация о объекте
    :param frame: исходный кадр видео
    :return: кадр с размеченным объектом
    """
    x, y, width, height = object_info['coordinates']
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(frame, object_info['class'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def draw_parking_area(frame):
    """
    Разметка парковочной зоны на кадре видео
    :param frame: исходный кадр видео
    :return: кадр с размеченной парковочной зоной
    """
    #разметка с цветом
    #pts = np.array(parking_area, np.int32)
    #cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    #return frame

    pts = np.array(parking_area, np.int32)
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [pts], (0, 255, 0))
    alpha = 0.0001  # Прозрачность разметки (задайте значение от 0 до 1)
    frame = cv2.addWeighted(frame, 1 - alpha, mask, alpha, 0)
    return frame

def draw_road_area(frame):
    """
    Разметка дорожной зоны на кадре видео
    :param frame: исходный кадр видео
    :return: кадр с размеченной дорожной зоной
    """
    #разметка с цветом
    # pts = np.array(road_area, np.int32)
    # cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
    # return frame

    pts = np.array(road_area, np.int32)
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [pts], (0, 0, 255))
    alpha = 0.00001  # Прозрачность разметки (задайте значение от 0 до 1)
    frame = cv2.addWeighted(frame, 1 - alpha, mask, alpha, 0)
    return frame


# Открытие видео файла или захват видео с камеры
video_capture = cv2.VideoCapture(r'C:\Users\ADMIN\PycharmProjects\pythonProject\video4.mp4')

# Здесь может быть указан путь к видео файлу вместо 0 для захвата видео с камеры

# Размеры сетки
grid_size = 50

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Cursor: ({x}, {y})")
        cv2.putText(frame, f"Cursor: ({x}, {y})", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv2.namedWindow('Object Detection')
cv2.setMouseCallback('Object Detection', mouse_callback)


if __name__ == "__main__":
    while True:
        # Чтение кадра видео
        ret, frame = video_capture.read()

        # Применение YOLO для распознавания объектов и разметка парковочной зоны
        output_frame = apply_yolo_object_detection(frame)

        # Разметка сетки
        for x in range(0, frame.shape[1], grid_size):
            cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 0, 0), 1)
        for y in range(0, frame.shape[0], grid_size):
            cv2.line(frame, (0, y), (frame.shape[1], y), (255, 0, 0), 1)

        # Отображение результата
        cv2.imshow('Object Detection', output_frame)

        # Прерывание цикла при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    video_capture.release()
    cv2.destroyAllWindows()