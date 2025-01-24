# Crop Disease Detection System

## Contents
* Project Synopsis
* Tools
    * platform
    * sensor
    * software
    * dataset
* Big Picture
* Project Details
    * AI model
    * flask server
    * raspberry pi
    * Integaration
* Demo
* Review


## Project Synopsis
* 프로젝트 명 : 농작물 병해충 관리 AI 시스템
* 진행 기간 : 2025.01.13 ~ 2025.01.24
* 목적 : 객체 탐지 기술을 활용하여 농작물의 병해충 감염 유무를 탐지하고, 이를 휴대폰 어플리케이션 및 스프링쿨러 등을 이용하여 농장 관리 시스템을 구축해 장기간 외출시에도 보다 편리한 농장 관리를 가능하게 하기 위함을 목적으로 하였습니다.
* 내용 : 라즈베리 파이 환경에서 OpenCv를 통한 이미지 처리, tensorflow를 이용한 농작물 병해충 탐지 모델 구축, flask 서버 구축 및 flask 서버와 모델을 이용한 농작물 병해충 감염 유무 판단, app inventer를 활용한 flask 서버에서 정보를 받아 스마트폰에서 확인할 수 있는 어플리케이션 제작을 하였고, 이를 잘 동작하는지 시연해 보았습니다.
* 결과
   * 라즈베리파이 환경에서 OpenCv를 통해 이미지를 캡쳐하고 저장할 수 있었습니다.
   * tensorflow를 이용하여 사과의 병해충 탐지 모델을 구축할 수 있었습니다.
   * 사과 병해충 탐지 모델을 포함한 flask 서버를 구축하여 사과의 병해충 감염 유무를 판단해볼 수 있었습니다. 
   * app inventer를 활용하여 스마트폰 어플리케이션을 제작할 수 있었습니다.(flask 서버 정보 불러오기 실패)
   * 라즈베리파이 환경에서 캡쳐하고 저장한 이미지를 서버로 전송하여 병해충 감염 유무를 판단할 수 있다는 것을 확인해볼 수 있었습니다.


 ## Tools
 ### Platform
 * Rasberry Pi 4B
 ### Sensor
 * USB WebCam
 ### Software
 * vscode


 ## Big Picture
 * 농작물의 병해충을 탐지하기 위해 모델을 만들고, 해당 모델을 포함한 서버를 구축하여 라즈베리파이에서 캡쳐한 이미지를 서버로 전송되게 하였고, 이를 통하여 농작물의 병해충을 탐지할 수 있도록 하였습니다.


 ## Project Details
 ### AI model

 kaggle에서 [augmented apple datasets](https://www.kaggle.com/datasets/rm1000/augmented-apple-disease-detection-dataset "kaggle")를 다운받아 이를 활용하여 AI model을 구축하였습니다. AI model을 구축 할 때에는 vscode를 이용하여 코드를 작성하였으며, 모델 구축시 tensorflow를 활용하였습니다. 

 * 파이썬 코드
 ```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# 데이터 경로 설정
train_dir = "C:/data/aaa/train"          # 훈련 데이터 경로
validation_dir = "C:/data/aaa/validation"  # 유효성 데이터 경로

# 데이터 전처리 (정규화만 적용)
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # 이미지 크기 통일
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# 클래스 수 확인
num_classes = len(train_generator.class_indices)
print(f"Detected {num_classes} classes: {train_generator.class_indices}")

# CNN 모델 생성 (정규화 및 드롭아웃 추가)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # 클래스 수에 맞게 출력층 생성
])

# 모델 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 조기 종료 콜백 설정
early_stopping = EarlyStopping(
    monitor='val_loss',  # 검증 손실 기준
    patience=5,          # 성능 개선이 없으면 5 에포크 후 종료
    restore_best_weights=True
)

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50,  # 최대 50 에포크 설정
    callbacks=[early_stopping]
)

# 학습 결과 시각화
plt.figure(figsize=(12, 4))

# Accuracy 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Loss 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

# 모델 저장
model.save("C:/datamodel/apple_model.h5")
```


 * 모델 학습 결과 그래프
![Image](https://github.com/user-attachments/assets/7cf1fc96-b5ea-4f22-8876-4f4d7b92bc9a)


 ### Flask server

 윈도우 환경에서 코드를 작성하여 윈도우 명령창을 통해 flask server를 실행시켰습니다. flask server는 AI model이 포함되게 작성했으며, 해당 서버에서 라즈베리파이에서 찍은 사진을 전송받고, 전송받은 사진을 server에 포함된 AI model을 이용하여 병해충 감염 유무를 판별하게 하였습니다. 또한, flask 서버를 구축한 환경과 다른 wifi에 연결되어있어도 라즈베리파이에서 flask 서버에 접근할 수 있도록 ngrok을 이용하여 서버의 접근성을 높였습니다.


* 파이썬 코드
 ```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 모델 로드
try:
    model = tf.keras.models.load_model(r"C:/datamodel/apple_model.h5")  # 모델 경로 설정
    print("모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 로드 에러: {e}")
    model = None

# 이미지 전처리 함수
def preprocess_image(image):
    try:
        target_size = (128, 128)  # 모델 입력 크기 (128x128으로 수정)
        image = image.resize(target_size)  # 크기 조정
        image = np.array(image) / 255.0  # 정규화

        # 흑백 이미지 처리
        if len(image.shape) == 2:  # 흑백 이미지일 경우
            image = np.expand_dims(image, axis=-1)  # 흑백 채널 추가
            image = np.repeat(image, 3, axis=-1)  # RGB로 변환

        # 모델 입력 형태로 변환
        image = np.expand_dims(image, axis=0)  # 배치 차원 추가
        return image
    except Exception as e:
        raise ValueError(f"이미지 전처리 에러: {e}")

@app.route("/")
def home():
    return "Flask 서버가 실행 중입니다!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")  # 이미지를 RGB로 변환
        processed_image = preprocess_image(image)  # 이미지 전처리
        prediction = model.predict(processed_image)  # 예측 수행

        # 병해충 여부 판단
        is_infected = prediction[0][0] > 0.5  # 0번째 출력 노드 값으로 병해충 여부 판단
        result = "병해충 의심" if is_infected else "좋음"

        # 결과 반환
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

* 서버 사진

![Image](https://github.com/user-attachments/assets/cdb560cf-d099-45ab-a7c9-0334a0646f1a)


 ### Rasberry Pi

 USB WebCam을 연결하여 OpenCv를 활용해 이미지를 3시간에 한 번씩 캡쳐하고, 이를 flask 서버로 전송하게 했습니다.

 ```python
import cv2
import time
from datetime import datetime
import requests

# Flask 서버의 Ngrok URL
FLASK_SERVER_URL ="https://95c1-210-93-56-120.ngrok-free.app/predict"  # Ngrok URL 입력

# USB 웹카메라 초기화
camera = cv2.VideoCapture(0)  # 기본 USB 웹캠 '/dev/video0'
if not camera.isOpened():
    print("USB 웹카메라를 찾을 수 없습니다.")
    exit()

try:
    while True:
        # 사진 촬영
        ret, frame = camera.read()
        if ret:
            # 현재 날짜와 시간을 기반으로 파일 이름 생성
            filename = f"captured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)  # 촬영된 이미지를 저장
            print(f"{filename} 저장 완료")

            # 저장된 이미지를 Flask 서버로 전송
            with open(filename, "rb") as img_file:
                files = {"file": img_file}
                try:
                    response = requests.post(FLASK_SERVER_URL, files=files)
                    if response.status_code == 200:
                        result = response.json().get("result", "알 수 없는 결과")
                        print(f"Flask 서버 응답: {result}")
                    else:
                        print(f"Flask 서버 오류: {response.status_code}, {response.text}")
                except Exception as e:
                    print(f"Flask 서버 전송 중 오류: {e}")

        else:
            print("카메라에서 이미지를 가져오지 못했습니다.")

        # 3시간(10800초) 대기
        time.sleep(10800)

except KeyboardInterrupt:
    print("프로그램 종료")
finally:
    camera.release()
```


 ### App

 App Inventer를 활용하여 스마트폰 어플리케이션을 제작했습니다. 앱과 서버의 연동이 잘 이루어지지 않아 앱에서 병해충 탐지 결과를 확인 하는 것에는 실패하였지만, flask 서버에서 App Inventer가 처리하기 쉽게 응답을 JSON 형식으로 변환해주면 문제가 해결될 것이라고 생각합니다.


* 앱 화면 구성
  
![Image](https://github.com/user-attachments/assets/212516c2-7c3f-497b-b2df-a9b2c0b3623e)


* 앱 블록코딩 구성
  
![Image](https://github.com/user-attachments/assets/f0d4716d-f4e5-45b1-9d01-fdb3fb03a68d)


 ### Intergration

 사과 병해충 데이터셋을 이용해 제작한 AI model을 flask 서버에 포함시키고, 라즈베리파이와 연결된 USB WebCam에서 캡쳐한 이미지를 서버로 전송시켜 서버에서는 해당 이미지를 판별하게 했으므로 AI model, flask server, Rasberry Pi, USB WebCam을 모두 연결시켰다고 볼 수 있습니다.


 ## Demo

 https://github.com/user-attachments/assets/a561fc51-2b57-4f9a-b4a6-ba76a35f652a


 ## Review
 * 본 프로젝트에서 라즈베리파이, USB WebCam, kaggle의 무료 데이터셋을 통해 농작물 병해충 감지 AI 시스템을 구현해낼 수 있었습니다. 이번 활동을 통해 AI model을 제작하는 방법, flask 서버를 구축하는 방법 등에 대해 배웠고, 이를 직접 구축해봄으로써 농장에 더 직접적인 도움을 줄 수 있는 시스템으로 발전시켜나가고 싶다는 생각을 하게 되었습니다.
 * 본 프로젝트가 2주간 진행되는 것이다 보니 첫 기획 단계보다 많이 간결한 결과물을 도출하게 되었지만, 실제 농장에 도움을 줄 수 있는 시스템으로 발전시키기 위해서는 GPU를 활용하여 더 많은 데이터셋을 학습시켜 AI model을 강화시키고, 농작물 사진 촬영은 더욱 좋은 화질과 넓은 폭으로 촬영할 수 있는 카메라를 이용하고, flask server에서 판별한 결과로는 물이나 해충제를 분사시킬 수 있는 스프링쿨러를 자동으로 제어할 뿐만 아니라 판별 결과를 스마트폰 어플리케이션으로 전송하여 어플리케이션 내에서 농작물 사진 확인 및 병해충 탐지 결과를 확인할 수 있도록 발전시킬 필요가 있다고 생각합니다.
