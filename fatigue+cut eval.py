from imutils import face_utils
import imutils
import dlib
import torch
import cv2

import numpy as np
import librosa
import sounddevice as sd
import time
from threading import Thread    

# 검출된 부분만 남기고 전부 흰색으로 변경
def extract_contours(frame):
   # RGB(255,0,0) 색상 범위 정의 (OpenCV는 BGR 순서를 사용하므로 (0, 0, 255))
   lower_green = np.array([0, 255, 0])
   upper_green = np.array([0, 255, 0])
   lower_red = np.array([255, 0, 0])
   upper_red = np.array([255, 0, 0])
 
   rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   # 정의한 색상 범위에 따라 마스크 생성
   mask = cv2.inRange(rgb, lower_green, upper_green)
   mask1 = cv2.inRange(rgb, lower_red, upper_red)

   mask = cv2.bitwise_or(mask, mask1)

   # 마스크를 사용하여 원본 이미지에서 색상을 추출
   res = cv2.bitwise_and(frame, frame, mask=mask)

   # 추출된 색상을 제외한 나머지 부분을 흰색으로 만듦
   res[np.where((res == [0,0,0]).all(axis=2))] = [255, 255, 255]
 
   return res

# 가이드라인 표시
def draw_guidelines(frame, width, height):
   cv2.rectangle(frame, (width//3 + 50, height//6 + 120, 25, 20), (128, 128, 254), 2)
   cv2.rectangle(frame, (width//3 + 95, height//6 + 120, 25, 20), (128, 128, 254), 2)
   cv2.rectangle(frame, (width//3 + 65, height//6 + 175, 45, 20), (128, 128, 254), 2)



# 알림 소리를 출력하는 클래스
class Sound():
   def __init__(self, file_path):
      # 음원 데이터 배열, 샘플링 속도
      self.y, self.sr = librosa.load(file_path)

   def play_music(self, duration):   
      sd.play(self.y[:int(self.sr*duration)], self.sr)  
      sd.wait()
   

class CNN(torch.nn.Module):
   def __init__(self):
      super(CNN, self).__init__()  
      
      # 첫번째 합성층
      # 필터 : 3 x 3, 입력 이미지 : 640 x 480 x 3, 출력 이미지 : 320 x 240 x 32
      self.layer1 = torch.nn.Sequential(
         torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
         torch.nn.ReLU(),
         torch.nn.MaxPool2d(kernel_size=2, stride=2))

      # 두번째 합성층
      # 필터 : 3 x 3, 입력이미지 : 640 x 480 x 3, 출력 이미지 : 160 x 120 x 64
      
      self.layer2 = torch.nn.Sequential(
         torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
         torch.nn.ReLU(),
         torch.nn.MaxPool2d(kernel_size=2, stride=2))

      # 전결합층 
      # stride가 2인 Max Pooling을 2번 수행했으므로 원본 이미지의 크기를 4로 나눔
      self.fc = torch.nn.Linear(640 // 4 * 480 // 4 * 64, 3, bias=True)

      # 전결합층 한정으로 가중치 초기화 
      torch.nn.init.xavier_uniform_(self.fc.weight)
      
   def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
      out = self.fc(out)
      self.out = out
      return out
      
    
    
# 자원 저장 경로
DATA_URI = './data/'   ##### 본인 환경에 맞게 변경
NORMAL_FACE_PATH = DATA_URI + 'normal'
SLEEP_EYE_PATH = DATA_URI + 'sleep_eye'
SLEEP_MOUTH_PATH = DATA_URI + 'sleep_mouth'
TEST_PATH = DATA_URI + 'test'

CURRENT_PATH = NORMAL_FACE_PATH   ##### 사진 종류 변경 시 변수만 변경


# face detection 모델
predictor_path = './vsc/ai/shape_predictor_68_face_landmarks.dat'

# 미리 학습된 모델 파일 경로
MODEL_PATH = DATA_URI + 'weights/30-7/ver1/model_state_dict.pt'
MODEL_PATH = './data_bak2/weights/30-7/ver1/model_state_dict.pt'



beep = Sound(DATA_URI + 'sound/beep.wav')


# 이미지 분류 함수 정의
def classify_image(model, image):
   # 이미지를 텐서로 변환
   image_tensor = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
   
   # 모델에 입력하여 예측 수행
   with torch.no_grad():
      outputs = model(image_tensor)
      
   # 예측 결과 중 확률이 가장 높은 클래스 선택
   _, predicted = torch.max(outputs, 1)

   return predicted.item()  # 클래스 인덱스 반환

# 전방 미주시 타이머
# 전방을 주시하고 있지 않을 시 부저 알림
def eye_road_timer():
   global face_detected, eye_road_thread_created
   print("Eyes on the road")
   while not face_detected:
      time.sleep(0)
      beep.play_music(0.5)
   
   eye_road_thread_created = False


# 눈 감음 타이머
# 눈 감긴 시간이 2초가 넘어갈 경우 부저 알림
def blink_timer():
   global now, blink_count, classification_result, blink_thread_created
   start = now

   while classification_result == 1:
      time.sleep(0)
      blink_count = now - start
      if blink_count > 2:
         print("Time to sleep")
         beep.play_music(1)
         break

   blink_count = 0
   blink_thread_created = False

   
# 하품 카운터
# 하품 횟수가 1분 안에 2회가 넘어갈 경우 부저 알림
def yawn_timer():
   global now, yawn_count, yawn_thread_created
   
   start = now
   print(yawn_count)
   while now - start < 60:
      time.sleep(0)
      if yawn_count >= 2:
         print("you are tired")
         beep.play_music(2)
         break
      
   yawn_count = 0
   yawn_thread_created = False
   
 
# 기준 타이머
def start_timer():
   global now
   while True:
      time.sleep(0)
      now = int(time.time())
      

   
def main():
   # 기준 타이머 초기화
   timer_thread = Thread(target=start_timer, daemon=True)
   timer_thread.start() 

   # 모델 불러오기
   model = CNN()
   model.load_state_dict(torch.load(MODEL_PATH))
   model.eval()

   # 웹캠 초기화
   camera = cv2.VideoCapture(0)
   if not camera.isOpened():
      raise ConnectionError('카메라 연결에 실패했습니다.\n카메라 연결을 확인해주세요.')
   width=int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
   height=int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))


   # dlib의 face landmark detector 초기화
   detector = dlib.get_frontal_face_detector()
   predictor = dlib.shape_predictor(predictor_path)

   # 각 눈, 입 랜드마크의 인덱스 불러오기
   # 왼눈 : 37 ~ 42, 오른눈 : 43 ~ 48, 입 : 48 ~ 68, 얼굴 윤곽 : 0 ~ 27
   (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
   (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
   (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
   (fStart, fEnd) = (0, 27)


   # 변수 초기화
   # 스레드가 접근해야 하는 전역 변수
   global blink_count, yawn_count, classification_result, face_detected
   global eye_road_thread_created, blink_thread_created, yawn_thread_created

   # 화면에 출력할 메시지 변수
   msg = ""
   color = 0
   
   # 얼굴 감지 관련 변수
   classification_result = 0
   blink_count = 0
   yawn_count = 0
   face_detected = False
   face_detect_timer = now
   face_detect_interval = 0
   yawnning = False
 
   # 스레드 Mutex 변수
   eye_road_thread_created = False
   blink_thread_created = False
   yawn_thread_created = False
 
   # 초기값(프레임) 버리기
   ret, frame = camera.read()
   if not ret:
      raise RuntimeError("프레임을 받아올 수 없습니다.\n카메라 연결을 확인해주세요.")
   while True: 
      # 얼굴 감지를 위해 프레임을 그레이스케일로 변경
      ret, frame = camera.read()
      if not ret:
         raise RuntimeError("프레임을 받아올 수 없습니다.\n카메라 연결을 확인해주세요.")

      frame = imutils.resize(frame, width=640)
      frame = cv2.flip(frame, 1)
      origin = frame.copy()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
      # 그레이스케일 영상으로 부터 얼굴 감지
      rects = detector(gray, 0)
  
      # face landmark 표시
      for rect in rects:
         # 얼굴 영역의 랜드마크 좌표를 계산하고 numpy array로 변환
         shape = predictor(gray, rect)
         shape = face_utils.shape_to_np(shape)

         # 눈, 입, 얼굴 윤곽의 좌표를 계산
         leftEye = shape[lStart:lEnd]
         rightEye = shape[rStart:rEnd]
         mouth = shape[mStart:mEnd]
         faceOutline = shape[fStart:fEnd]
         
         
         # 눈과 입, 얼굴 윤곽의 볼록 껍질 표시
         leftEyeHull = cv2.convexHull(leftEye)
         rightEyeHull = cv2.convexHull(rightEye)
         mouthHull = cv2.convexHull(mouth)
         faceOutlineHull = cv2.convexHull(faceOutline)
         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
         cv2.drawContours(frame, [rightEyeHull], -1, (0,255, 0), 1)
         cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
         cv2.drawContours(frame, [faceOutlineHull], -1, (0, 255, 0), 1)


      # 눈, 입, 얼굴 윤곽의 볼록 껍질 라벨링 및 가이드라인 표시
      labelled_frame = extract_contours(frame)
      draw_guidelines(labelled_frame, width, height)


      # 이미지를 CNN 모델에 전달하여 분류 수행
      if now - face_detect_timer >= face_detect_interval:
         classification_result = classify_image(model, labelled_frame)
         face_detect_timer = now


      # 분류 결과에 따라 적절한 후속 작업 수행
      face_detected = True if len(rects) > 0 else False
  
      if not face_detected:             # 전방을 주시하지 않는 상태 
         msg = "Alert : Eyes on the road!!"
         color = (63, 51, 230)

         if eye_road_thread_created == False:
            eye_road_thread_created = True
            eye_road_thread = Thread(target=eye_road_timer, daemon=True)
            eye_road_thread.start()
   
      elif classification_result == 0:  # 일반 상태
         msg = "Normal"      
         color = (83, 229, 72)
         yawnning = False
   
      elif classification_result == 1:  # 눈 감은 상태
         msg = "Warning : Eye closed!!"
         color = (14, 101, 230)
         
         yawnning = False
   
         # 타이머 측정하는 스레드 생성
         if blink_thread_created is False:
            blink_thread_created = True
            blink_thread = Thread(target=blink_timer, daemon=True)
            blink_thread.start()
   
      elif classification_result == 2:  # 하품하는 상태
         msg = "Caution : Yawning!!"
         color = (76, 228, 230)

         if yawnning is False:
            yawnning = True
            yawn_count += 1
            
            # 카운트 측정하는 스레드 생성
            if yawn_thread_created is False:
               yawn_thread_created = True
               yawn_thread = Thread(target=yawn_timer, daemon=True)
               yawn_thread.start()
      

      cv2.putText(origin, msg, (10, 55),cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
      cv2.imshow("Origin", origin)
      cv2.imshow("labelled", labelled_frame)
  
      key = cv2.waitKey(1) & 0xFF
  
      if key == ord("q"):
         break

   # finalize
   cv2.destroyAllWindows()
   camera.release()


if __name__ == '__main__':
   main()