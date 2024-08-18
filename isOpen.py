# Các chỉ số điểm mắt trái và phải
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