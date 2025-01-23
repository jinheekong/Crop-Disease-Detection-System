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
 *
