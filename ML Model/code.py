import os
from datetime import datetime
from PIL import Image, ImageDraw
import cv2
from datetime import datetime, timezone, timedelta
import requests
import json
import re
from tqdm import tqdm
from decouple import config
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

def ocr_space_file(filename, overlay, api_key, language):
    payload = {
                'isOverlayRequired': overlay,
                'apikey': api_key,
                'language': language,
                'OCREngine': 2,
            }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    data = json.loads(r.content.decode())

    lines = data["ParsedResults"][0]["TextOverlay"]["Lines"]

    lpnum = "".join(line["LineText"] for line in lines)
    lpnum = re.sub(r'[^a-zA-Z0-9]', '', lpnum)
    
    return lpnum

def draw_detections(p1, p2, p3, p4, img):
    class_colors = {
        'helmet': 'blue',
        'motorcyclist': 'green',
        'license_plate': 'red',
        'face': 'darkmagenta',
        'front': 'darkgoldenrod',
        'rear': 'darkorchid',
        'mobile': 'orange',
        'wheelie': 'purple'
    }
    
    draw = ImageDraw.Draw(img)

    preds = {'predictions': p1['predictions'] + p2['predictions'] + p3['predictions'] + p4['predictions']}

    for prediction in preds['predictions']:
        x, y, width, height = (
            prediction['x'],
            prediction['y'],
            prediction['width'],
            prediction['height']
        )
        
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        
        class_name = prediction['class']
        confidence = prediction['confidence']
        
        label_color = class_colors.get(class_name, 'black')

        if class_name=='motorcyclist':
            draw.rectangle([x1, y1, x2, y1+14], fill=label_color)
            label_position = (x1 + 5, y1 + 2)
        else:
            draw.rectangle([x1, y1-14, x2, y1], fill=label_color)
            label_position = (x1 + 5, y1-12)
            
        draw.rectangle([x1, y1, x2, y2], outline=label_color, width=2)

        label = f"{class_name} ({confidence:.2f})"
        draw.text(label_position, label, fill='white')

    return img

# Roboflow API keys
roboflow_api_key = config("ROBOFLOW_API_KEY")

# Configure clients with different confidence thresholds
custom_configuration1 = InferenceConfiguration(confidence_threshold=0.8, iou_threshold=0.4)
client1 = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api_key
)
client1.configure(custom_configuration1)

custom_configuration2 = InferenceConfiguration(confidence_threshold=0.4, iou_threshold=0.3)
client2 = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api_key
)
client2.configure(custom_configuration2)

custom_configuration3 = InferenceConfiguration(confidence_threshold=0.1, iou_threshold=0.1)
client3 = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api_key
)
client3.configure(custom_configuration3)

# New client for mobile and wheelie detection
custom_configuration4 = InferenceConfiguration(confidence_threshold=0.4, iou_threshold=0.4)
client4 = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api_key
)
client4.configure(custom_configuration4)

video_path = 'input2.mp4'
cap = cv2.VideoCapture(video_path)
# Get video details
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(5)
total_frames = int(cap.get(7))

# Violate Date folder
current_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%d-%m-%Y")
folder_path = os.path.join(os.getcwd(), f"Violations/{current_date}")
os.makedirs(folder_path, exist_ok=True)
violations = []
# Process every 30th frame
for frame_number in tqdm(range(0, total_frames, 30), desc="Processing frames", unit="frames"):
    ret, frame = cap.read()
    if not ret:
        break

    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    image_path = "temp_frame.jpg"
    pil_frame.save(image_path)

    r1 = client1.infer(image_path, model_id="helmet-detection-project/13")
    r11 = client1.infer(image_path, model_id="traffic-detection-sutq6/36")
    pred1 = r1['predictions']
    pred11 = r11['predictions']
    print(pred1)
    
    for pr1 in pred1:
        helmet_detected = False
        face_detected = False
        rear_detected = False
        more_than_two_detected = False
        mobile_detected = False
        wheelie_detected = False
        num_faces_detected = 0
        num_helmets_detected = 0

        if pr1['class'] == 'motorcyclist':
            motorcyclist_x, motorcyclist_y, motorcyclist_width, motorcyclist_height = pr1['x'], pr1['y'], pr1['width'], pr1['height']
            
            motorcyclist_x1, motorcyclist_y1 = int(motorcyclist_x - motorcyclist_width / 2), int(motorcyclist_y - motorcyclist_height / 2)
            motorcyclist_x2, motorcyclist_y2 = int(motorcyclist_x + motorcyclist_width / 2), int(motorcyclist_y + motorcyclist_height / 2)
            
            motorcyclist_image = pil_frame.crop((motorcyclist_x1, motorcyclist_y1, motorcyclist_x2, motorcyclist_y2))
            motorcyclist_image.save("temp_motorcyclist_image.jpg")

            # Lane check
            r3 = client3.infer("temp_motorcyclist_image.jpg", model_id="two-wheeler-lane-detection/3")
            lane = r3

            if lane['predictions']:
                max_conf = max(lane['predictions'], key=lambda x: x['confidence'])
                lane['predictions'] = [max_conf]
            
            pred3 = lane['predictions']
            
            for lane_prediction in pred3:
                if lane_prediction['class'] == 'rear':
                    rear_x, rear_y, rear_width, rear_height = lane_prediction['x'], lane_prediction['y'], lane_prediction['width'], lane_prediction['height']

                    if motorcyclist_x1 < rear_x < motorcyclist_x2 and motorcyclist_y1 < rear_y < motorcyclist_y2:
                        rear_detected = True
                        break

            # Face detected
            r2 = client2.infer("temp_motorcyclist_image.jpg", model_id="face-detection-mik1i/21")
            pred2 = r2['predictions']

            for face_prediction in pred2:
                if face_prediction['class'] == 'face':
                    face_x, face_y, face_width, face_height = face_prediction['x'], face_prediction['y'], face_prediction['width'], face_prediction['height']

                    if motorcyclist_x1 < face_x < motorcyclist_x2 and motorcyclist_y1 < face_y < motorcyclist_y2:
                        num_faces_detected += 1

                        # Avoid detecting helmet and face in same area and calculating number of people incorrectly
                        for helmet_prediction in pred1:
                            if helmet_prediction['class'] == 'helmet':
                                helmet_x, helmet_y, helmet_width, helmet_height = helmet_prediction['x'], helmet_prediction['y'], helmet_prediction['width'], helmet_prediction['height']
                                
                                face_x1 = face_x - face_width / 2
                                face_y1 = face_y - face_height / 2
                                face_x2 = face_x + face_width / 2
                                face_y2 = face_y + face_height / 2

                                helmet_x1 = helmet_x - helmet_width / 2
                                helmet_y1 = helmet_y - helmet_height / 2
                                helmet_x2 = helmet_x + helmet_width / 2
                                helmet_y2 = helmet_y + helmet_height / 2

                                overlap_x1 = max(face_x, helmet_x)
                                overlap_y1 = max(face_y, helmet_y)
                                overlap_x2 = min(face_x + face_width, helmet_x + helmet_width)
                                overlap_y2 = min(face_y + face_height, helmet_y + helmet_height)

                                overlap_width = max(0, overlap_x2 - overlap_x1)
                                overlap_height = max(0, overlap_y2 - overlap_y1)

                                overlap_area = overlap_width * overlap_height

                                face_area = face_width * face_height

                                if overlap_area / face_area > 0.6:
                                    num_faces_detected -= 1
                                    break

            if num_faces_detected > 0:
                face_detected = True

            # Helmet check
            for helmet_prediction in pred1:
                if helmet_prediction['class'] == 'helmet':
                    helmet_x, helmet_y, helmet_width, helmet_height = helmet_prediction['x'], helmet_prediction['y'], helmet_prediction['width'], helmet_prediction['height']

                    if motorcyclist_x1 < helmet_x < motorcyclist_x2 and motorcyclist_y1 < helmet_y < motorcyclist_y2:
                        helmet_detected = True
                        num_helmets_detected += 1

            # More than two riding
            if num_faces_detected + num_helmets_detected > 2:
                more_than_two_detected = True

            # New mobile and wheelie detection
            r4 = client4.infer("temp_motorcyclist_image.jpg", model_id="tvd-kp9qw/2")
            pred4 = r4['predictions']

            for violation_prediction in pred4:
                if violation_prediction['class'] == 'mobile':
                    mobile_x, mobile_y, mobile_width, mobile_height = violation_prediction['x'], violation_prediction['y'], violation_prediction['width'], violation_prediction['height']
                    
                    if motorcyclist_x1 < mobile_x < motorcyclist_x2 and motorcyclist_y1 < mobile_y < motorcyclist_y2:
                        mobile_detected = True
                        break
                
                if violation_prediction['class'] == 'wheelie':
                    wheelie_x, wheelie_y, wheelie_width, wheelie_height = violation_prediction['x'], violation_prediction['y'], violation_prediction['width'], violation_prediction['height']
                    
                    if motorcyclist_x1 < wheelie_x < motorcyclist_x2 and motorcyclist_y1 < wheelie_y < motorcyclist_y2:
                        wheelie_detected = True
                        break

            # Colored motorcycle image with all detections
            colored_motorcycle = draw_detections(r1, r2, lane, r4, motorcyclist_image)
            
            # Violated license plate
            if not helmet_detected or face_detected or rear_detected or more_than_two_detected or mobile_detected or wheelie_detected:
                
                violation_names = []
                if not helmet_detected or face_detected:
                    violation_names.append('no_helmet')
                if rear_detected:
                    violation_names.append('wrong_lane')
                if more_than_two_detected:
                    violation_names.append('triple_riding')
                if mobile_detected:
                    violation_names.append('mobile_usage')
                if wheelie_detected:
                    violation_names.append('one_wheeling')

                timestamp = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%d-%m-%Y %H %M %S")
                image_name = ", ".join(violation_names) + f" - {timestamp}"
                lp_detected = False

                for pr11 in pred1:
                    if pr11['class'] == 'license_plate':
                        license_plate_x, license_plate_y, license_plate_width, license_plate_height = pr11['x'], pr11['y'], pr11['width'], pr11['height']
                        if motorcyclist_x1 < license_plate_x < motorcyclist_x2 and motorcyclist_y1 < license_plate_y < motorcyclist_y2:
                            license_plate_x1, license_plate_y1 = int(license_plate_x - license_plate_width / 2), int(license_plate_y - license_plate_height / 2)
                            license_plate_x2, license_plate_y2 = int(license_plate_x + license_plate_width / 2), int(license_plate_y + license_plate_height / 2)
            
                            license_plate_image = pil_frame.crop((license_plate_x1, license_plate_y1, license_plate_x2, license_plate_y2))
                            
                            license_plate_image.save("temp_lp.jpg")
                            lpnum = ocr_space_file(filename="temp_lp.jpg", overlay=False, api_key=config("OCR_SPACE_API"), language='eng')   

                            if lpnum.strip():
                                image_name = lpnum + " - " + image_name
                            else:
                                image_name = image_name
                            image_folder_path = os.path.join(folder_path, image_name)
                            os.makedirs(image_folder_path, exist_ok=True)

                            violated_motorcycle_image_path = os.path.join(image_folder_path, f"{lpnum} - motorcyclist.jpg")
                            colored_motorcycle.save(violated_motorcycle_image_path)

                            violated_motorcycle_lp_image_path = os.path.join(image_folder_path, f"{lpnum} - license_plate.jpg")
                            license_plate_image.save(violated_motorcycle_lp_image_path)

                            lp_text_path = os.path.join(image_folder_path, f"{lpnum} - license_plate_number.txt")
                            with open(lp_text_path, 'w') as file:
                                file.write(f"Violated License Plate Number - {lpnum}")

                            lp_detected = True

                            if os.path.exists("temp_lp.jpg"):
                                os.remove("temp_lp.jpg")
                            break
                if not lp_detected:
                    image_folder_path = os.path.join(folder_path, image_name)
                    os.makedirs(image_folder_path, exist_ok=True)
                    violated_motorcycle_image_path = os.path.join(image_folder_path, f"motorcyclist.jpg")

                    colored_motorcycle.save(violated_motorcycle_image_path)
            else:
                print("No Violation Detected")
            # Violated license plate and other violations
            if not helmet_detected or face_detected or rear_detected or more_than_two_detected:
                violation_names = []
                if not helmet_detected or face_detected:
                    violations.append("No Helmet")
                    # violation_names.append('no_helmet')
                if rear_detected:
                    violations.append("Wrong Lane")
                    # violation_names.append('wrong_lane')
                if more_than_two_detected:
                    violations.append("Triple Riding")
                    # violation_names.append('triple_riding')
                if mobile_detected:
                    violations.append("Mobile Usage")
                    # violation_names.append('mobile_usage')
                if wheelie_detected:
                    violations.append("One Wheeling")
                    # violation_names.append('one_wheeling')

                # print(f"\nViolations detected: {', '.join(violation_names)}")
                
                timestamp = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%d-%m-%Y %H %M %S")
                image_name = ", ".join(violation_names) + f" - {timestamp}"
                lp_detected = False
        else:
            print("Video is Garbage")

current_datetime = datetime.now()
current_date = current_datetime.strftime("%Y-%m-%d")
current_time = current_datetime.strftime("%H:%M:%S")
violations = set(violations)
if len(violations) > 0:
    status = "Valid"
else:
    status = "Garbage"
print("****************Report Summary********************")
print(f"Name: Syed Sarib Naveed\nID: BAD-112\nVideo Status: {status}\nDate: {current_date}\nTime: {current_time}")
if status == 'Valid':
    print(f"Violations Details:")
    for violation in violations:
        print(violation)

if os.path.exists("temp_motorcyclist_image.jpg"):
    os.remove("temp_motorcyclist_image.jpg")
cap.release()

print("*********************End************************")