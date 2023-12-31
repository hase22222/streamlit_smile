import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from io import  BytesIO, BufferedReader


#st.write('自分の顔をカメラで撮影してみましょう！')
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(static_image_mode = True, max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5)

upload_img = st.file_uploader("画像をアップロードしましょう！", type = ['png', 'jpg'])


if upload_img is not None:
    st.image(upload_img)
    bytes_data = upload_img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img)
    height, width, _ = img.shape  # 画像の寸法を取得
    if not results.multi_face_landmarks:
        st.write('顔が検出できないようです。再取り込みを行ってください。')
    else:
        for face_landmarks in results.multi_face_landmarks:
            # 目と鼻のランドマーク
            left_eye = face_landmarks.landmark[362] # 左目
            right_eye = face_landmarks.landmark[133] # 右目
            nose_tip = face_landmarks.landmark[1] # 鼻先

            # ランドマークのY座標
            left_eye_y = int(left_eye.y * height)
            right_eye_y = int(right_eye.y * height)
            nose_tip_y = int(nose_tip.y * height)

        # 目と鼻のY座標がほぼ同じか判定
        if abs(left_eye_y - right_eye_y) <= 4.0 and abs(nose_tip_y - ((left_eye_y + right_eye_y) / 2)) >= 30:
            height, width, _ = img.shape  # 画像の寸法を取得

            # 座標をピクセル単位に変換
            left_eye_cornerx  = int(results.multi_face_landmarks[0].landmark[130].x * width)     # 左目尻（x座標）
            left_eye_cornery  = int(results.multi_face_landmarks[0].landmark[130].y * height)    # 左目尻（y座標）
            right_eye_cornerx  = int(results.multi_face_landmarks[0].landmark[263].x * width)    # 右目尻（x座標）
            right_eye_cornery  = int(results.multi_face_landmarks[0].landmark[263].y * height)   # 右目尻（y座標）
            left_mouth_cornerx  = int(results.multi_face_landmarks[0].landmark[61].x * width)    # 左口角（y座標）
            left_mouth_cornery  = int(results.multi_face_landmarks[0].landmark[61].y * height)   # 左口角（y座標
            right_mouth_cornerx  = int(results.multi_face_landmarks[0].landmark[291].x * width)  # 右口角（x座標）  
            right_mouth_cornery  = int(results.multi_face_landmarks[0].landmark[291].y * height) # 右口角（y座標）
            upper_ripx = int(results.multi_face_landmarks[0].landmark[13].x * width)             # 上唇  （x座標）
            upper_ripy = int(results.multi_face_landmarks[0].landmark[13].y * height)            # 上唇  （y座標）

            h_length = right_eye_cornerx - left_eye_cornerx    # 右目尻 - 左目尻
            v_length1 = left_mouth_cornery - left_eye_cornery  # 左口角 - 左目尻
            v_length2 = right_mouth_cornery - left_eye_cornery # 右口角 - 右目尻

            #cv2.line(img, (right_eye_cornerx, right_eye_cornery), (right_eye_cornerx, right_mouth_cornery), (255, 0, 0), 2)
            #cv2.line(img, (left_eye_cornerx, left_eye_cornery), (left_eye_cornerx, left_mouth_cornery), (255, 0, 0), 2)
            #cv2.line(img, (right_eye_cornerx, right_eye_cornery), (left_eye_cornerx, left_eye_cornery), (255, 0, 0), 2)        
            #cv2.line(img, (right_eye_cornerx, right_mouth_cornery), (left_eye_cornerx, left_mouth_cornery), (255, 0, 0), 2)
            #cv2.line(img, (right_eye_cornerx, right_mouth_cornery), (upper_ripx, upper_ripy), (255, 0, 0), 2)    

        #比率を計算
            ratio1 = h_length / v_length1  
            ratio2 = h_length / v_length2

        # ①：右口角と左口角の位置が、上唇より上昇している
            if (upper_ripy > left_mouth_cornery) & (upper_ripy > right_mouth_cornery):
              # ②：　左右の目尻間の幅:（目尻 - 口角) = 約1.63:1　が黄金比
              if (1.6 <= ratio1 <= 1.7) & (1.6 <= ratio2 <= 1.7):
                  state = 'これは素晴らしい笑顔ですね！！' 
              elif (1.3 <= ratio1 <=2.0) & (1.3 <= ratio2 <= 2.0):
                  state = '良い笑顔ですね！　もっと笑ってみましょう！'
              else:
                  state = '本物の笑顔ではないですね!'
            else:
                state = 'もっと笑いましょう!'

            output_img = img.copy()
            ret, enco_img = cv2.imencode(".png", cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))

            BytesIO_img = BytesIO(enco_img.tostring())
            BufferedReader_img = BufferedReader(BytesIO_img)

            output_img2 = img.copy()
            ret2, enco_img2 = cv2.imencode(".png", cv2.cvtColor(output_img2, cv2.COLOR_BGR2RGB))

            BytesIO_img2 = BytesIO(enco_img2.tostring())
            BufferedReader_img2 = BufferedReader(BytesIO_img2)

            
            st.markdown('<style>h1{font-size: 40px; text-align: center;}</style>', unsafe_allow_html=True)
            st.markdown('<h1>👇</h1>', unsafe_allow_html=True)
            st.markdown('<style>h2{font-size: 40px;}</style>', unsafe_allow_html=True)
            st.markdown('<h2>判定結果</h2>', unsafe_allow_html=True)
            st.markdown('<style>h3{font-size: 20px;}</style>', unsafe_allow_html=True)
            st.markdown(f'<h3>{state}</h3>', unsafe_allow_html=True)
            st.image(output_img)
        else:
           st.text("顔が正面を向いていません。再取り込みを行ってください。")