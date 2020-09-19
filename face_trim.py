import cv2
import time

# 폴더에서 이미지파일을 읽어서 하나씩 가능하도록 list append 형식으로 변경 예정  
# 이미지 테스트는 10개에 이미지의 각도를 변경시킨 이미지를 추가 
# 이미지들로 충분한 정확도를 도출할 수 있을 지 판단
# CNN으로 진행 계획 

# start time
start = time.time() 

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image 
img = cv2.imread("./test_image/test1.jpeg")
print(img)
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    img_trim = img[y:y+h, x:x+w]
    cv2.imwrite('./result_image/org_trim.jpg', img_trim)
    org_image = cv2.imread('./result_image/org_trim.jpg')
# Display the output

# cv2.imshow('img', img_trim)
# cv2.waitKey()

# compile end time 
print(time.time()-start)