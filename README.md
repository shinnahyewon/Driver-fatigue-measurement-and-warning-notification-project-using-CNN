# Driver-fatigue-measurement-and-warning-notification-project-using-CNN
Driver fatigue measurement and warning notification project using CNN

수행기간 : 2024.02.13.~2024.02.22

담당역할 : - 커스텀 데이터셋 제작
          - 얼굴 인식 가이드라인 제작
          - CNN 학습 코드 서브

수행목표 : -이미지 인식을 통해 보호구역에 진입시 자동으로 속도 조절
          - 얼굴인식과 전처리 과정을 거친 이미지를 데이터로 가공하여 CNN로 특징을 추출
          - CNN 학습된 모델을 실시간 영상에 적용하여 운전자의 상태를 분석

사용기술 : Python, PyTorch, OpenCV, Dlib

세부수행내용 : 

■ 프로젝트 개요
1) 개발 배경
 최근 몇 년 간 봄철(3~5월) 졸음운전으로 인한 교통사고가 하루 마다 평균 7건 발생했다.
이번 프로젝트를 통해 운전자의 피로도를 측정하고 경고함으로써 졸음운전 발생사고를 예방하고자 한다.

![image](https://github.com/shinnahyewon/Driver-fatigue-measurement-and-warning-notification-project-using-CNN/assets/161293023/d2d8a6de-496e-48e6-b7dd-1e3618991871)

2) 개발 목표
 - 얼굴인식과 전처리 과정을 거친 이미지를 데이터로 가공하여 CNN로 특징을 추출
 - CNN 학습된 모델을 실시간 영상에 적용하여 운전자의 상태를 분석


   
■ 시스템 구성

![image](https://github.com/shinnahyewon/Driver-fatigue-measurement-and-warning-notification-project-using-CNN/assets/161293023/6d9eca01-2ccd-46c4-8b7e-e8dd0fc72247)

1. Normal 평상시 운전자 상태
2. Eye Closed상태 – 1초마다 blink count+1
3. Yawing상태 – 1회 하품시 yawn count +1


4. ■ 기대 효과
- 운전자의 상태를 분석해 졸음운전으로 일어날 사고를 예방.
