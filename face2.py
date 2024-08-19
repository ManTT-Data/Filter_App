from operator import rshift
import cv2 as cv 
import numpy as np
import mediapipe as mp
import cvzone
import filter
from isOpen import isOpen

mp_face_mesh = mp.solutions.face_mesh

overlay_image = cv.imread('image_no_bg.png', cv.IMREAD_UNCHANGED)

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def animated(pic, x):
    if pic < 119:
        overlay_image = cv.imread(f'{x}/{pic}.jpg', cv.IMREAD_UNCHANGED)
    if pic < 119:
        pic += 1
    return overlay_image, pic

def mask_eyes(frame, frame_raw, mesh_points):
    # Tạo mặt nạ cho mí mắt trái và phải
    mask_left_eye = np.zeros_like(frame[:, :, 0])
    mask_right_eye = np.zeros_like(frame[:, :, 0])
    
    # Vẽ vùng mắt màu đen
    cv.fillPoly(mask_left_eye, [mesh_points[LEFT_EYE]], 255)
    cv.fillPoly(mask_right_eye, [mesh_points[RIGHT_EYE]], 255)
    
    # Tạo mặt nạ toàn màu trắng
    left_eye = np.ones_like(frame[:, :, 0]) * 255
    right_eye = np.ones_like(frame[:, :, 0]) * 255

    # Vẽ vùng mắt màu đen
    cv.fillPoly(left_eye, [mesh_points[LEFT_EYE]], 0)
    cv.fillPoly(right_eye, [mesh_points[RIGHT_EYE]], 0)
    
    # Tạo mặt nạ tổng hợp cho cả hai mắt
    mask_eyes = cv.bitwise_or(mask_left_eye, mask_right_eye)
    eyes = cv.bitwise_and(left_eye, right_eye)
    
    # Áp dụng mặt nạ để che phần ảnh bị mí mắt che phủ
    masked_frame = cv.bitwise_and(frame, frame, mask=mask_eyes)
    without_eyes = cv.bitwise_or(frame_raw, frame_raw, mask = eyes)
    final = cv.bitwise_or(without_eyes, masked_frame)

    return final

def display_video():
    filter.pic = 0

    # Khởi tạo giá trị alpha (độ trong suốt)
    initial_alpha = 0.0       # Độ trong suốt ban đầu (0 là hoàn toàn trong suốt)

    # Giá trị thay đổi qua từng khung hình
    alpha = initial_alpha
    final_alpha = 1.0         # Độ trong suốt cuối cùng
    alpha_increment = 0.02   # Tăng alpha theo từng khung hình

    video_path = './video_test/video_test.mp4'
    cap = cv.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_raw = cv.flip(frame, 1)
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # Chọn filter
                if filter.mangekyou == True:
                    x = 's2'
                else: 
                    x = 's1'
                if filter.pic < 119:
                    overlay_image, filter.pic = animated(filter.pic, x)

                mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                
                # Kiểm tra xem mắt trái, phải có mở không
                left_eye_open = isOpen(frame, mesh_points, 'LEFT EYE', 2.5)
                right_eye_open = isOpen(frame, mesh_points, 'RIGHT EYE', 2.5)

                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                # Chèn ảnh vào mống mắt 
                overlay_l = cv.resize(overlay_image, (int(1.8 * l_radius), int(1.8 * l_radius)))
                overlay_r = cv.resize(overlay_image, (int(1.8 * r_radius), int(1.8 * r_radius)))
                
                # Tách kênh alpha ra từ ảnh overlay
                if overlay_l.shape[2] == 4:
                    alpha_channel_l = overlay_l[:, :, 3]  # Kênh alpha của overlay trái
                    overlay_l = overlay_l[:, :, :3]  # Kênh RGB của overlay trái
                    alpha_channel_r = overlay_r[:, :, 3]  # Kênh alpha của overlay phải
                    overlay_r = overlay_r[:, :, :3]  # Kênh RGB của overlay phải

                    # Điều chỉnh kênh alpha
                    alpha_channel_l = (alpha_channel_l * alpha).astype(np.uint8)
                    alpha_channel_r = (alpha_channel_r * alpha).astype(np.uint8)

                    # Tạo mặt nạ dựa trên kênh alpha mới
                    overlay_l_with_alpha = cv.merge([overlay_l[:, :, 0], 
                                                      overlay_l[:, :, 1], 
                                                      overlay_l[:, :, 2], 
                                                      alpha_channel_l])

                    overlay_r_with_alpha = cv.merge([overlay_r[:, :, 0], 
                                                      overlay_r[:, :, 1], 
                                                      overlay_r[:, :, 2], 
                                                      alpha_channel_r])
                else:
                    overlay_l_with_alpha = overlay_l
                    overlay_r_with_alpha = overlay_r

                # Điều chỉnh alpha cho khung hình tiếp theo
                alpha = min(final_alpha, alpha + alpha_increment) # Đảm bảo alpha không vượt quá 1
                
                # Chèn ảnh sử dụng mask
                if left_eye_open:
                    frame = cvzone.overlayPNG(frame, overlay_l_with_alpha, 
                                                        [int(l_cx - overlay_l_with_alpha.shape[1] // 2), 
                                                        int(l_cy - overlay_l_with_alpha.shape[0] // 2)])
                if right_eye_open:
                    frame = cvzone.overlayPNG(frame, overlay_r_with_alpha, 
                                                        [int(r_cx - overlay_r_with_alpha.shape[1] // 2), 
                                                        int(r_cy - overlay_r_with_alpha.shape[0] // 2)])
                # Optimize filter
                final = mask_eyes(frame, frame_raw, mesh_points)
            else:
                filter.pic = 0
                filter.sharingan = False
    
            if filter.sharingan == True:
                cv.imshow('img', final)
            else: 
                cv.imshow('img', frame_raw)

            key = cv.waitKey(10)
            if key == ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()
