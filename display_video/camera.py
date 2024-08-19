import cv2 as cv 
import numpy as np
import mediapipe as mp 
import cvzone
import filter as filter
from display_video import filter_effect

mp_face_mesh = mp.solutions.face_mesh

# Points in the eye area and iris
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def display_video():
    filter.pic = 0

    # Transparency
    alpha = 0.0

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
                    x = 'mangekyou_sharingan'
                else: 
                    x = 'sharingan'
                if filter.pic < 119:
                    overlay_image, filter.pic = filter_effect.animated(filter.pic, x)

                mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                
                # Check if left and right eyes are open
                left_eye_open = filter_effect.isOpen(frame, mesh_points, 'LEFT EYE', 2.5)
                right_eye_open = filter_effect.isOpen(frame, mesh_points, 'RIGHT EYE', 2.5)

                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                # Resize overlay
                overlay_l = cv.resize(overlay_image, (int(2 * l_radius), int(2 * l_radius)))
                overlay_r = cv.resize(overlay_image, (int(2 * r_radius), int(2 * r_radius)))

                # Create transparency for filter effect
                alpha, overlay_l_with_alpha, overlay_r_with_alpha = filter_effect.transparency(overlay_l, overlay_r, alpha, final_alpha = 1.0, alpha_increment = 0.02)

                # Fill overlay_image by using mask
                if left_eye_open:
                    frame = cvzone.overlayPNG(frame, overlay_l_with_alpha, 
                                                        [int(l_cx - overlay_l_with_alpha.shape[1] // 2), 
                                                        int(l_cy - overlay_l_with_alpha.shape[0] // 2)])
                if right_eye_open:
                    frame = cvzone.overlayPNG(frame, overlay_r_with_alpha, 
                                                        [int(r_cx - overlay_r_with_alpha.shape[1] // 2), 
                                                        int(r_cy - overlay_r_with_alpha.shape[0] // 2)])
                # Create the final image
                final = filter_effect.mask_eyes(frame, frame_raw, mesh_points)
            else:
                filter.pic = 0
                # filter.sharingan = False
    
            if filter.sharingan == True:
                cv.imshow('img', final)
            else: 
                cv.imshow('img', frame_raw)

            key = cv.waitKey(1)
            if key ==ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()