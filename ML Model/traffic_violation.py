import os
from datetime import datetime
import geocoder
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
custom_configuration4 = InferenceConfiguration(confidence_threshold=0.8, iou_threshold=0.4)
client4 = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api_key
)
client4.configure(custom_configuration4)

# video_path = 'input5.mp4'
# cap = cv2.VideoCapture(video_path)
# # Get video details
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fps = cap.get(5)
# total_frames = int(cap.get(7))

# # Violate Date folder
# current_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%d-%m-%Y")
# folder_path = os.path.join(os.getcwd(), f"Violations/{current_date}")
# os.makedirs(folder_path, exist_ok=True)
# violations = []
# # Process every 30th frame
# for frame_number in tqdm(range(0, total_frames, 30), desc="Processing frames", unit="frames"):
#     ret, frame = cap.read()
#     if not ret:
#         break

#     pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     image_path = "temp_frame.jpg"
#     pil_frame.save(image_path)

#     r1 = client1.infer(image_path, model_id="traffic-detection-sutq6/36")
#     r2 = client2.infer(image_path, model_id="driver-detection/1")
#     pred1 = r1['predictions']
#     pred2 = r2['predictions']
#     if len(pred1) != 0 or len(pred2) != 0:
#         r2 = client2.infer(image_path, model_id="tvd-kp9qw/2")
#         r3 = client3.infer(image_path, model_id="license-plate-recognition-rxg4e/6")
#         r4 = client4.infer(image_path, model_id="helmet-detection-project/13")
#         pred2 = r2['predictions']
#         pred3 = r3['predictions']
#         pred4 = r4['predictions']
#         license_plate_number = "Not Found"
#         if len(pred4) != 0:
#             if pred4[0]['class'] == 'motorcyclist':
#                 violations.append("No Helmet")
#         if len(pred2) != 0:
#             violations.append(pred2[0]['class'])
#             # Extract license plate number if present
#             if len(pred3) > 0:
#                 license_plate_bbox = pred3[0]
#                 x, y, width, height = (
#                     license_plate_bbox['x'],
#                     license_plate_bbox['y'],
#                     license_plate_bbox['width'],
#                     license_plate_bbox['height']
#                 )
#                 # Crop the license plate area
#                 lp_x1 = int(x - width / 2)
#                 lp_y1 = int(y - height / 2)
#                 lp_x2 = int(x + width / 2)
#                 lp_y2 = int(y + height / 2)
#                 cropped_lp = frame[lp_y1:lp_y2, lp_x1:lp_x2]
#                 cropped_lp_image = Image.fromarray(cropped_lp)
#                 cropped_lp_image.save("temp_license_plate.jpg")

#                 # Perform OCR on cropped license plate
#                 license_plate_number = ocr_space_file(
#                     "temp_license_plate.jpg",
#                     overlay=False,
#                     api_key=config("OCR_SPACE_API"),
#                     language="eng"
#                 )

# # Generate Report
# current_datetime = datetime.now()
# current_date = current_datetime.strftime("%Y-%m-%d")                      
# current_time = current_datetime.strftime("%H:%M:%S")
# # Fetch the current location using geocoder
# g = geocoder.ip('me')  
# location = g.latlng 
# # Get the address
# if location:
#     latitude, longitude = location
#     address = g.address if g.address else "Unknown Location"
# else:
#     latitude, longitude = None, None
#     address = "Unknown"

# violations = set(violations)
# if len(violations) > 0:
#     status = "Valid"
# else:
#     status = "Garbage"

# print("\n****************Report Summary********************")
# print(f"Citizen ID:LA-7431\nCitizen Name: Syed Sarib Naveed\nReporting ID: BAD-112\nVideo Status: {status}\nDate: {current_date}\nTime: {current_time} \nLocation: {address}")
# if status == 'Valid':
#     print(f"\nViolations Details:")
#     for violation in violations:
#         print(violation)
#         print(f"Associated License Plate: {license_plate_number}")  # Print license plate for each violation

# if os.path.exists("temp_motorcyclist_image.jpg"):
#     os.remove("temp_motorcyclist_image.jpg")
# if os.path.exists("temp_license_plate.jpg"):
#     os.remove("temp_license_plate.jpg")
# cap.release()

# print("*********************End************************")


import os
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
from decouple import config
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import requests
import json
import re

# (Existing code)

# def process_video(video_path):
#     # Initialize required variables
#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     fps = cap.get(5)
#     total_frames = int(cap.get(7))

#     # Violate Date folder
#     current_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%d-%m-%Y")
#     folder_path = os.path.join(os.getcwd(), f"Violations/{current_date}")
#     os.makedirs(folder_path, exist_ok=True)
#     violations = []
#     license_plate_number = "Not Found"

#     # Process every 30th frame
#     for frame_number in tqdm(range(0, total_frames, 30), desc="Processing frames", unit="frames"):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         image_path = "temp_frame.jpg"
#         pil_frame.save(image_path)

#         # Perform inference and detection
#         r1 = client1.infer(image_path, model_id="traffic-detection-sutq6/36")
#         r2 = client2.infer(image_path, model_id="driver-detection/1")
#         pred1 = r1['predictions']
#         pred2 = r2['predictions']
#         if len(pred1) != 0 or len(pred2) != 0:
#             r2 = client2.infer(image_path, model_id="tvd-kp9qw/2")
#             r3 = client3.infer(image_path, model_id="license-plate-recognition-rxg4e/6")
#             r4 = client4.infer(image_path, model_id="helmet-detection-project/13")
#             pred2 = r2['predictions']
#             pred3 = r3['predictions']
#             pred4 = r4['predictions']
#             if len(pred4) != 0 and pred4[0]['class'] == 'motorcyclist':
#                 violations.append("No Helmet")
#             if len(pred2) != 0:
#                 violations.append(pred2[0]['class'])
#             if len(pred3) > 0:
#                 license_plate_bbox = pred3[0]
#                 x, y, width, height = (
#                     license_plate_bbox['x'],
#                     license_plate_bbox['y'],
#                     license_plate_bbox['width'],
#                     license_plate_bbox['height']
#                 )
#                 lp_x1 = int(x - width / 2)
#                 lp_y1 = int(y - height / 2)
#                 lp_x2 = int(x + width / 2)
#                 lp_y2 = int(y + height / 2)
#                 cropped_lp = frame[lp_y1:lp_y2, lp_x1:lp_x2]
#                 cropped_lp_image = Image.fromarray(cropped_lp)
#                 cropped_lp_image.save("temp_license_plate.jpg")
#                 license_plate_number = ocr_space_file(
#                     "temp_license_plate.jpg",
#                     overlay=False,
#                     api_key=config("OCR_SPACE_API"),
#                     language="eng"
#                 )

#     # Prepare a structured response
#     status = "Valid" if violations else "Rejected"
#     current_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%H:%M:%S")
    
#     response = {
#         "location": "Mianwali, Pakistan",  # or populate with real location if available
#         "violation_type": ", ".join(set(violations)) if violations else "No violation",
#         "confidence": 85,  # You can adjust this to a real confidence score if you have one
#         "status":status,
#         "details": {
#             "Citizen ID": "LA-7431",
#             "Citizen Name": "Syed Sarib Naveed",
#             "Reporting ID": "BAD-112",
#             "Video Status": status,
#             "Date": current_date,
#             "Time": current_time,
#             "Violations": list(set(violations)),
#             "License Plate": license_plate_number,
#         }
#     }

#     # Cleanup temporary files
#     if os.path.exists("temp_frame.jpg"):
#         os.remove("temp_frame.jpg")
#     if os.path.exists("temp_license_plate.jpg"):
#         os.remove("temp_license_plate.jpg")

#     # Release resources
#     cap.release()
#     return response


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)
    total_frames = int(cap.get(7))

    # Violation Date folder
    current_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%d-%m-%Y")
    folder_path = os.path.join(os.getcwd(), f"Violations/{current_date}")
    os.makedirs(folder_path, exist_ok=True)
    violations = []
    license_plate_number = "Not Found"
    violation_confidences = []

    # Process every 30th frame
    for frame_number in tqdm(range(0, total_frames, 30), desc="Processing frames", unit="frames"):
        ret, frame = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_path = "temp_frame.jpg"
        pil_frame.save(image_path)

        # Perform inference and detection
        r1 = client1.infer(image_path, model_id="traffic-detection-sutq6/36")
        r2 = client2.infer(image_path, model_id="driver-detection/1")
        pred1 = r1['predictions']
        pred2 = r2['predictions']
        if len(pred1) != 0 or len(pred2) != 0:
            r2 = client2.infer(image_path, model_id="tvd-kp9qw/2")
            r3 = client3.infer(image_path, model_id="license-plate-recognition-rxg4e/6")
            r4 = client4.infer(image_path, model_id="helmet-detection-project/13")
            pred2 = r2['predictions']
            pred3 = r3['predictions']
            pred4 = r4['predictions']
            
            # Check for violations
            if len(pred4) != 0 and pred4[0]['class'] == 'motorcyclist':
                violations.append("No Helmet")
                violation_confidences.append(pred4[0]['confidence'])
            
            if len(pred2) != 0:
                violations.append(pred2[0]['class'])
                violation_confidences.append(pred2[0]['confidence'])
            
            if len(pred3) > 0:
                license_plate_bbox = pred3[0]
                x, y, width, height = (
                    license_plate_bbox['x'],
                    license_plate_bbox['y'],
                    license_plate_bbox['width'],
                    license_plate_bbox['height']
                )
                lp_x1 = int(x - width / 2)
                lp_y1 = int(y - height / 2)
                lp_x2 = int(x + width / 2)
                lp_y2 = int(y + height / 2)
                cropped_lp = frame[lp_y1:lp_y2, lp_x1:lp_x2]
                cropped_lp_image = Image.fromarray(cropped_lp)
                cropped_lp_image.save("temp_license_plate.jpg")
                license_plate_number = ocr_space_file(
                    "temp_license_plate.jpg",
                    overlay=False,
                    api_key=config("OCR_SPACE_API"),
                    language="eng"
                )

    # Calculate average confidence
    average_confidence = sum(violation_confidences) / len(violation_confidences) if violation_confidences else 0

    # Prepare a structured response
    status = "Valid" if violations else "Rejected"
    current_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%H:%M:%S")
    
    response = {
        "location": "Mianwali, Pakistan",  # or populate with real location if available
        "violation_type": ", ".join(set(violations)) if violations else "No violation",
        "average_confidence": round(average_confidence * 100, 2),  # Confidence as percentage
        "status": status,
        "details": {
            "Citizen ID": "LA-7431",
            "Citizen Name": "Syed Sarib Naveed",
            "Reporting ID": "BAD-112",
            "Video Status": status,
            "Date": current_date,
            "Time": current_time,
            "Violations": list(set(violations)),
            "License Plate": license_plate_number,
        }
    }

    # Cleanup temporary files
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")
    if os.path.exists("temp_license_plate.jpg"):
        os.remove("temp_license_plate.jpg")

    # Release resources
    cap.release()
    return response
