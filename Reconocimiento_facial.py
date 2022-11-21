import cv2
import face_recognition

#Importamos la imagen
image = cv2.imread("Imagenes/Nicolas.jpg")
face_loc = face_recognition.face_locations(image)[0]
#print("imagen:", face_loc)
face_image_econdings = face_recognition.face_encodings(image, known_face_locations=[face_loc])[0]
#print("face_image_encodings:", face_image_econdings)

# #Imprimir imagenes
# cv2.rectangle(image, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 255, 25))
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Video streaming
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
     ret, frame = cap.read()
     if ret == False: break
     frame = cv2.flip(frame, 1)

     face_locations = face_recognition.face_locations(frame)
     if face_locations != []:
          for face_location in face_locations:
               face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]
               result = face_recognition.compare_faces([face_image_econdings], face_frame_encodings)

               #print("Result:", result)
               if result[0] == True:
                    text = "Nico"
                    color = (125, 220, 0)
               else:
                    text = "Desconocido"
                    color = (50, 50, 255)
                    
               cv2.rectangle(frame, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), color, -1)
               cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
               cv2.putText(frame, text, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)

     cv2.imshow("Video", frame)
     k = cv2.waitKey(1)
     if k == 27 & 0xFF:
          break
cap.release()
cv2.destroyAllWindows()