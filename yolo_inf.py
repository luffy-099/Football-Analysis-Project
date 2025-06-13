from ultralytics import YOLO

model = YOLO("/Users/madalenacontente/Desktop/Projeto_APVC/models/best.pt")  # Load small version

results = model.predict("/Users/madalenacontente/Desktop/Projeto_APVC/VSBraga/trim1.mp4",save=True)
print(results[0])  # Print results to consol
print('=====================================================================')
for box in results[0].boxes:
    print(box)
