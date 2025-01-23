# Disease and (Insect)Pest Detection System

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

 kaggle에서 agument apple datasets를 다운받아 이를 활용하여 AI model을 구축하였습니다. AI model을 구축 할 때에는 vscode를 이용하여 코드를 작성하였으며, 모델 구축시 tensorflow를 활용하였습니다. 

 ### Flask server

 윈도우 환경에서 코드를 작성하여 윈도우 명령창을 통해 flask server를 실행시켰습니다. flask server는 AI model이 포함되게 작성했으며, 해당 서버에서 라즈베리파이에서 찍은 사진을 전송받고, 전송받은 사진을 server에 포함된 AI model을 이용하여 병해충 감염 유무를 판별하게 하였습니다. 또한, flask 서버를 구축한 환경과 다른 wifi에 연결되어있어도 라즈베리파이에서 flask 서버에 접근할 수 있도록 ngrok을 이용하여 서버의 접근성을 높였습니다.

 ### Rasberry Pi

 USB WebCam을 연결하여 OpenCv를 활용해 이미지를 3시간에 한 번씩 캡쳐하고, 이를 flask 서버로 전송하게 했습니다.

 ### Intergration

 사과 병해충 데이터셋을 이용해 제작한 AI model을 flask 서버에 포함시키고, 라즈베리파이와 연결된 USB WebCam에서 캡쳐한 이미지를 서버로 전송시켜 서버에서는 해당 이미지를 판별하게 했으므로 AI model, flask server, Rasberry Pi, USB WebCam을 모두 연결시켰다고 볼 수 있습니다.

 ## Demo

 ## Review
 * 본 프로젝트에서 라즈베리파이, USB WebCam, kaggle의 무료 데이터셋을 통해 농작물 병해충 감지 AI 시스템을 구현해낼 수 있었습니다. 이번 활동을 통해 AI model을 제작하는 방법, flask 서버를 구축하는 방법 등에 대해 배웠고, 이를 직접 구축해봄으로써 농장에 더 직접적인 도움을 줄 수 있는 시스템으로 발전시켜나가고 싶다는 생각을 하게 되었습니다.
 * 본 프로젝트가 2주간 진행되는 것이다 보니 첫 기획 단계보다 많이 간결한 결과물을 도출하게 되었지만, 실제 농장에 도움을 줄 수 있는 시스템으로 발전시키기 위해서는 GPU를 활용하여 더 많은 데이터셋을 학습시켜 AI model을 강화시키고, 농작물 사진 촬영은 더욱 좋은 화질과 넓은 폭으로 촬영할 수 있는 카메라를 이용하고, flask server에서 판별한 결과로는 물이나 해충제를 분사시킬 수 있는 스프링쿨러를 자동으로 제어할 뿐만 아니라 판별 결과를 스마트폰 어플리케이션으로 전송하여 어플리케이션 내에서 농작물 사진 확인 및 병해충 탐지 결과를 확인할 수 있도록 발전시킬 필요가 있다고 생각합니다.
