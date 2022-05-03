import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

##### Pose addition
mp_pose = mp.solutions.pose # Initialize pose class
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) # Setup for video processing

def detectPose(img,pose_fxn):
    #original_img = img.copy()
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = pose_fxn.process(rgb_img)

    return result

def filter_shoulders_and_hands(pose_landmarks):
    '''
    11 - Left shoulder     12 - Right shoulder
    13 - Left Elbow        14 - Right Elbow
    15 - Left Wrist        16 - Right Wrist
    23 - Left Hip          24 - Right Hip
    '''
    fl = []
    idx = 0
    for data_pt in pose_landmarks.landmark:
        if idx >= 11 and idx <= 16:
            fl.append({
                'x': data_pt.x, 'y':data_pt.y, 'z': data_pt.z, 'visibility':data_pt.visibility
            })
        elif idx ==  23 or idx == 24:
            fl.append({
                'x': data_pt.x, 'y':data_pt.y, 'z': data_pt.z, 'visibility':data_pt.visibility
            })
        idx+=1
    filtered_landmarks = fl
    return filtered_landmarks
    
def check_rapid_motion(lndmrk,prev_time,now_time,motion_time=None,prev_lndmrk=None):
    """ lndmark is the landmark object for either elbows or wrists """
    if motion_time:
        if abs(lndmrk.x - prev_lndmrk.x)*img_w>100:
            motion_time += now_time-prev_time
        elif abs(lndmrk.y - prev_lndmrk.y)*img_h>100:
            motion_time += now_time-prev_time
        else:
            motion_time=0
    else:
        if abs(lndmrk.x - prev_lndmrk.x)*img_w>100:
            motion_time = now_time-prev_time
        elif abs(lndmrk.y - prev_lndmrk.y)*img_h>100:
            motion_time = now_time-prev_time
        
            

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/jidnyesha/Desktop/PABI/DR/output2.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    success, image = cap.read()

    start = time.time()
    original_img = image
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    head_pose_results = face_mesh.process(image)
    pose_results = detectPose(image,pose_video)
    #print(pose_results)
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    ''' if head_pose_results.multi_face_landmarks:
        for face_landmarks in head_pose_results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        #print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec) '''
    
    filtered_landmrk=filter_shoulders_and_hands(pose_results.pose_landmarks)
    

    if pose_results.pose_landmarks:
        #for pose_lndmrks in pose_results.pose_landmarks:
        for idx,lm in enumerate(pose_results.pose_landmarks.landmark):
            if idx == 13 or idx ==14:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                if idx == 13: 
                    side = "Left"
                else: 
                    side = "Right"    
                #cv2.putText(image, side + "Elbow x: " + str(np.round(x,2)), (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                #cv2.putText(image, side + "Elbow y: " + str(np.round(y,2)), (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                print("Elbows "+side+ str(x)+'    '+str(y))
            elif idx == 15 or idx ==16:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                if idx == 15: 
                    side = "Left"
                else: 
                    side = "Right"    
                #cv2.putText(image, side + "Wrist x: " + str(np.round(x,2)), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                #cv2.putText(image, side + "Wrist y: " + str(np.round(y,2)), (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                print("Wrists "+side+ str(x)+'    '+str(y))
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list = pose_results.pose_landmarks , 
            connections = mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255,255,255),thickness=3,circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),thickness=2,circle_radius = 2)
        )
    
    ##
    ##
    end = time.time()
    time_elapsed = end 
    #print('Time '+ str(time_elapsed)+str(filtered_landmrk))
    out.write(image)
    cv2.imshow('Head Pose Estimation', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

