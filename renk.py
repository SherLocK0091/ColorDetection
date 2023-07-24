import cv2
import numpy as np

# Renk aralıkları (BGR formatında)
lower_red = np.array([0, 0, 100]) #Kırmızı rengin alt ve üst sınırlarını 
upper_red = np.array([50, 50, 255])#BGR (Mavi, Yeşil, Kırmızı) renk uzayında belirler. BGR formatı, OpenCV tarafından kullanılan renk uzayıdır.

lower_yellow = np.array([0, 100, 100])
upper_yellow = np.array([50, 255, 255])

lower_blue = np.array([100, 0, 0])
upper_blue = np.array([255, 50, 50])

# Video akışı aç
video_capture = cv2.VideoCapture('videos/newyork.mp4')  

while video_capture.isOpened():#
    # Bir kare oku
    ret, frame = video_capture.read()#
    if not ret:
        break

    # Kopya bir kare oluşturuluyor
    processed_frame = frame.copy()#

    # Görüntüyü BGR'den HSV'ye dönüştür
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#

    # Kırmızı rengi tespit et
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in red_contours:
        cv2.drawContours(processed_frame, [contour], -1, (0, 0, 255), 2)  # Kırmızı renkli nesneleri kırmızıyla işaretle

    # Sarı rengi tespit et
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in yellow_contours:
        cv2.drawContours(processed_frame, [contour], -1, (0, 255, 255), 2)  # Sarı renkli nesneleri sarıyla işaretle

    # Mavi rengi tespit et
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in blue_contours:
        cv2.drawContours(processed_frame, [contour], -1, (255, 0, 0), 2)  # Mavi renkli nesneleri maviyle işaretle

    # Orjinal videoyu ayrı bir pencerede göster
    cv2.imshow('newyork.mp4', frame)

    # İşlenmiş videoyu başka bir pencerede göster
    cv2.imshow('Color Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
