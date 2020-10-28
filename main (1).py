from PIL import Image
import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/user/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
src = cv2.imread("1233.jpg")
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(src_gray,1.1,3)

for x, y, w, h in faces:
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = src[y: y + h, x: x + w]
    face_gray = src_gray[y: y + h, x: x + w]
print(x,  y,  w,  h)


#이미지 크롭 코드
img = Image.open('1233.jpg')
crop_area = (x,  y,  x+w,  y+h)
       ##(가로시작점, 세로시작점, 가로범위, 세로범위)
       ##보통 x, y w, h 변수로 사용됨

       ### 근데 저코드는 범위가 아니라 xy좌표 두개인듯
       ### 얼굴선은 따지는걸로 봐선 크롭만 xy 좌표 두개쓰는듯, 인식파트는 높이 넓이가 맞음
       #### 시작점은 그대로 2차좌표는 합치는걸로 수정함. -> 성공!
crop_img = img.crop(crop_area)
crop_img.save("croptest1233.jpg")
crop_img.show()

""" 사진인식 코드로 변경함
필요모듈 : 파이선 파이참 오픈cv PIL
필요자료 : 한글이름 아닌 사진파일 -> PNG JPG 지원, GIF 미지원 
            opencv가 한글 파일명을 지원하지 않음.
            Ref)->https://zzsza.github.io/data/2018/01/23/opencv-1/

변경요망사항 : 
    1. 캐스케이드에 쓸 xml파일 : 딥러닝데이터 정리파일
        * 캐스케이드 : cv에서 지원해주는 얼굴인식 누적데이터(딥러닝데이터)
    xml은 opencv 깔때 같이 다운받아짐.
    cv재설치하면 파이선 모듈있는 주소를 밷는데,(본인의 경우 아나콘다/립/사이트패키지)
    거기가서 cv2폴더안의 xml주소를 끌어와 사용.
    얘는 쓸일이 많으므로 C:로 시작되는 절대주소로 변경바람.
    Ref)-> https://ponyozzang.tistory.com/597

    2. 5행, 17행, 26행의 사진파일명.
        -1. 5행 17행은 원본 파일명.
        상대주소로 코딩해놨는데 (py파일이랑) 같은폴더에 넣고 파일명만 바꾸면 쓸수있음.
        -2. 26행은 크롭한 파일명.
        이것도 상대주소니까 같은 폴더에 저장됨.
        둘다 절대주소로 고치면 여러개 돌릴수 있을 것 같음. range로 파일명변경도 일괄 할수 있을 것 같음. 

"""