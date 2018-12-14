import enum

class Indexing(enum.Enum):
        #left ear
        Left_top_ear = 0
        Left_middle_ear = 1

        Chin = 8

        #right ear
        Right_middle_ear = 15
        Right_top_ear = 16

        # left eyebrow
        Left_eyebrow_outer = 17
        Left_eyebrow_m3 = 18
        Left_eyebrow_middle = 19
        Left_eyebrow_m1 = 20
        Left_eyebrow_inner = 21

        # right eyebrow
        Right_eyebrow_outer = 26
        Right_eyebrow_m3 = 25
        Right_eyebrow_middle = 24
        Right_eyebrow_m1 = 23
        Right_eyebrow_inner = 22

        #left eye
        Left_eye_left_corner = 36
        Left_eye_right_corner = 39

        # right eye
        Right_eye_left_corner = 42
        Right_eye_right_corner = 45

        #nose
        Top_nose = 27
        Bottom_nose = 33
        Left_nostril = 31
        Right_nostril = 35

        #mouth
        Left_mouth_corner = 48
        Right_mouth_corner = 54
        Bottom_mouth_point = 57
        Middle_upper_lip = 51
        Left_upper_lip = 50
        Right_upper_lip = 52

