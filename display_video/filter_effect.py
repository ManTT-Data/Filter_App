import cv2 as cv 
import numpy as np
import mediapipe as mp 

# Points in the eye area and iris
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def getSize(image, face_landmarks, INDEXES):
    '''
    This function computes the width and height of the face part (e.g. eye, mouth)
    '''
    index_list = [face_landmarks[index] for index in INDEXES]
    min_x = min([landmark[0] for landmark in index_list])
    max_x = max([landmark[0] for landmark in index_list])
    min_y = min([landmark[1] for landmark in index_list])
    max_y = max([landmark[1] for landmark in index_list])
    width = max_x - min_x
    height = max_y - min_y
    
    return width, height, (int(min_x), int(min_y))

def isOpen(image, face_landmarks, face_part, threshold):
    '''
    This function checks whether an eye or mouth of the person(s) is open, utilizing its facial landmarks.
    '''
    # Determine the indexes based on the face part
    if face_part == 'LEFT EYE':
        INDEXES = LEFT_EYE
    elif face_part == 'RIGHT EYE':
        INDEXES = RIGHT_EYE
    else:
        return False
    
    # Get the height of the face part.
    _, height, _ = getSize(image, face_landmarks, INDEXES)
    
    # Get the height of the whole face.
    _, face_height, _ = getSize(image, face_landmarks, [i for i in range(0, 468)])  # Use all landmarks for the face height
    
    # Check if the face part is open.
    return (height/face_height)*100 > threshold

# Filter animation
def animated(pic, x):
    if pic < 119:
        overlay_image = cv.imread(f'filters/{x}/{pic}.jpg', cv.IMREAD_UNCHANGED)
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

def transparency(overlay_l, overlay_r, alpha, final_alpha = 1.0, alpha_increment = 0.02):
    # Extract alpha channel from overlay image
    if overlay_l.shape[2] == 4:
        alpha_channel_l = overlay_l[:, :, 3]  
        overlay_l = overlay_l[:, :, :3]  
        alpha_channel_r = overlay_r[:, :, 3] 
        overlay_r = overlay_r[:, :, :3] 

        # Adjust alpha channel
        alpha_channel_l = (alpha_channel_l * alpha).astype(np.uint8)
        alpha_channel_r = (alpha_channel_r * alpha).astype(np.uint8)

        # Create new mask base on alpha channel
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

    # Adjust alpha for continue frame
    alpha = min(final_alpha, alpha + alpha_increment)

    return alpha, overlay_l_with_alpha, overlay_r_with_alpha