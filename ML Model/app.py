from flask import Flask, request, jsonify
import traffic_violation  # Import your ML script here

app = Flask(__name__)

@app.route('/traffic_violation', methods=['POST'])
def process_video():
    print("Request received at /traffic_violation")
    if 'video' not in request.files:
        print("No video found in request")
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    video_path = "input.mp4"
    video_file.save(video_path)

    print(f"Video saved to {video_path}")

    try:
        response = traffic_violation.process_video("input5.mp4")
        print(f"\n\n************ Machine Learning Response ****************")
        print(f"Citizen ID: {response['details']['Citizen ID']}")
        print(f"Citizen Name: {response['details']['Citizen Name']}")
        print(f"Reporting ID: {response['details']['Reporting ID']}")
        print(f"Video Status: {response['details']['Video Status']}")
        print(f"Confidence: {response['average_confidence']}")
        print(f"Date: {response['details']['Date']}")
        print(f"Time: {response['details']['Time']}")
        print(f"Violations: {response['details']['Violations']}")
        print(f"License Plate: {response['details']['License Plate']}")
        print("********************************************************")
        # print(f"ML response: {response}")
        return jsonify(response)
    except Exception as e:
        print(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# response = traffic_violation.process_video('input5.mp4')
# print(f"\n\n************ Machine Learning Response ****************")
# print(f"Citizen ID: ADA-34")
# print(f"Citizen Name: Syed Sarib Naveed")
# print(f"Reporting ID: BAD-1123")
# print(f"Video Status: Valid")
# print(f"Date: 19/06/2025")
# print(f"Time: 10:56:43")
# print(f"Violations: ['No Helmet', 'Mobile Phone Using']")
# print(f"License Plate: Not Found")
# print("********************************************************")