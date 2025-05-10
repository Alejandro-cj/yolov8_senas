from ultralytics import YOLO
import cv2

# Carga tu modelo entrenado
model = YOLO("runs/detect/train9/weights/best.pt")  # ajusta la ruta si es diferente

# Abre la cámara (0 es la webcam por defecto)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza la predicción
    results = model(frame, show=True)

    # Espera por la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
