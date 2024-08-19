import cv2 as cv 
import numpy as np
import mediapipe as mp 
import cvzone
import filter
from isOpen import isOpen

mp_face_mesh = mp.solutions.face_mesh

overlay_image = cv.imread('image_no_bg.png', cv.IMREAD_UNCHANGED)

# Points in the eye area and iris
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Filter animation
def animated(pic, x):
    if pic < 119:
        overlay_image = cv.imread(f'{x}/{pic}.jpg', cv.IMREAD_UNCHANGED)
    if pic < 119:
        pic += 1
    return overlay_image, pic

# Create mask to fill overlay
def mask_eyes(frame, frame_raw, mesh_points):
    # Create mask for left and right iris
    mask_left_eye = np.zeros_like(frame[:, :, 0])
    mask_right_eye = np.zeros_like(frame[:, :, 0])
    
    cv.fillPoly(mask_left_eye, [mesh_points[LEFT_EYE]], 255)
    cv.fillPoly(mask_right_eye, [mesh_points[RIGHT_EYE]], 255)
    
    # Create a full white mask
    left_eye = np.ones_like(frame[:, :, 0]) * 255
    right_eye = np.ones_like(frame[:, :, 0]) * 255

    # Fill black into the eye area
    cv.fillPoly(left_eye, [mesh_points[LEFT_EYE]], 0)
    cv.fillPoly(right_eye, [mesh_points[RIGHT_EYE]], 0)

    
    # Create composite mask for both eyes
    mask_eyes = cv.bitwise_or(mask_left_eye, mask_right_eye)
    eyes = cv.bitwise_and(left_eye, right_eye)
    
    # Apply mask to complete filter layer
    masked_frame = cv.bitwise_and(frame, frame, mask=mask_eyes)
    without_eyes = cv.bitwise_or(frame_raw, frame_raw, mask = eyes)
    final = cv.bitwise_or(without_eyes, masked_frame)

    return final

def display_video():
    filter.pic = 0
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
                
                # Check if left and right eyes are open
                left_eye_open = isOpen(frame, mesh_points, 'LEFT EYE', 2.5)
                right_eye_open = isOpen(frame, mesh_points, 'RIGHT EYE', 2.5)

                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                # Resize overlay
                overlay_l = cv.resize(overlay_image, (int(2 * l_radius), int(2 * l_radius)))
                overlay_r = cv.resize(overlay_image, (int(2 * r_radius), int(2 * r_radius)))
                
                # Fill overlay_image by using mask
                if left_eye_open:
                    frame = cvzone.overlayPNG(frame, overlay_l, 
                                                        [int(l_cx - overlay_l.shape[1] // 2), 
                                                        int(l_cy - overlay_l.shape[0] // 2)])
                if right_eye_open:
                    frame = cvzone.overlayPNG(frame, overlay_r, 
                                                        [int(r_cx - overlay_r.shape[1] // 2), 
                                                        int(r_cy - overlay_r.shape[0] // 2)])
                # Create the final image
                final = mask_eyes(frame, frame_raw, mesh_points)
            else:
                filter.pic = 0
                filter.sharingan = False
    
            if filter.sharingan == True:
                cv.imshow('img', final)
            else: 
                cv.imshow('img', frame_raw)

            key = cv.waitKey(1)
            if key ==ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()