#  이것은 coco_friend_voice.py 파일이야

import speech_recognition as sr
from time import time
import google.generativeai as genai
from gtts import gTTS
import cam1


expression = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 음성을 인식하기 위한 recognizer 클래스를 초기화합니다.
recognizer = sr.Recognizer()

# Google Cloud API 키를 설정합니다.
GOOGLE_API_KEY = 'AIzaSyDlDI7RYx-lxCU9adYh3C2xtewQyoDIgZo'

# generativeai 라이브러리에 Google Cloud API 키를 설정합니다.
genai.configure(api_key=GOOGLE_API_KEY)

# 마이크 설정을 시작합니다.
while True:  # 무한 루프를 시작합니다.
    with sr.Microphone() as source:  # 마이크를 열고 듣기를 시작합니다.
        print("1초 동안 준비")
        # 1초 동안 마이크를 열어서 주변 소음 수준을 확인합니다.
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("9초 동안 음성 듣기")
        # 9초 동안 마이크에서 음성을 듣고 인식합니다.
        audio = recognizer.listen(source, phrase_time_limit=9)
        
        # speak 함수를 정의합니다.
        def speak(text):
            # gTTS를 사용하여 입력된 텍스트를 음성으로 변환합니다.
            tts = gTTS(text=text, lang='ko')
            # 변환된 음성을 'voice.mp3' 파일로 저장합니다.
            tts.save('voice.mp3')    

    start = time()  # 음성 인식 시작 시간을 기록합니다.
    # 음성을 텍스트로 변환합니다.
    text = recognizer.recognize_google(audio, language='ko-KR')

    # Gemini에게 전달할 프롬프트를 설정합니다.
    prompt = f"지금{cam1.VAL1}감정을 참고해서 함께 3살 ~ 5살 처럼 간단하게 답변해줘"  
    print(f"{prompt}")
    print(f"나 : {text}")  # 사용자의 입력된 음성을 출력합니다.    
    
    # Gemini 모델을 생성합니다.
    model = genai.GenerativeModel('gemini-pro')    
    # 사용자의 입력을 포함한 프롬프트로 Gemini 모델에게 응답을 생성하도록 요청합니다.
    response = model.generate_content(prompt + text)
    
    # 응답이 존재하는 경우
    if response.candidates:
        # 응답을 합칩니다.
        result = ''.join([p.text for p in response.candidates[0].content.parts])  
    else:
        # 응답이 없는 경우
        result = "No candidates found"  
    
    print("Gemini: ", result)  # Gemini의 응답을 출력합니다.
    speak(result)  # Gemini의 응답을 TTS로 변환하고 'voice.mp3' 파일로 저장합니다.
