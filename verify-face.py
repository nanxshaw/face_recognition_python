from flask import Flask, request, jsonify
import face_recognition
import os
import logging

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

# Path absolut ke folder user images
BASE_USER_IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'boldeaccess-backend', 'assets', 'users'))

@app.route("/verify-face", methods=["POST"])
def verify_face():
    filename = request.form.get("user_image_name")
    if 'image' not in request.files or not filename:
        logging.error("Image file or filename missing in request")
        return jsonify({"match": False, "message": "Image or filename missing"}), 400

    uploaded_file = request.files['image']
    logging.info(f"Request received for user image: {filename}")

    try:
        uploaded_image = face_recognition.load_image_file(uploaded_file)
        logging.info("Uploaded image loaded successfully")
    except Exception as e:
        logging.error(f"Error loading uploaded image: {e}")
        return jsonify({"match": False, "message": f"Error loading uploaded image: {str(e)}"}), 500

    known_face_path = os.path.join(BASE_USER_IMAGE_DIR, filename)
    logging.info(f"Looking for user image at: {known_face_path}")

    if not os.path.exists(known_face_path):
        logging.error("User image not found at path")
        return jsonify({"match": False, "message": "User image not found"}), 404

    try:
        known_image = face_recognition.load_image_file(known_face_path)
        logging.info("Known user image loaded successfully")
    except Exception as e:
        logging.error(f"Error loading user image: {e}")
        return jsonify({"match": False, "message": f"Error loading user image: {str(e)}"}), 500

    try:
        known_encodings = face_recognition.face_encodings(known_image)
        uploaded_encodings = face_recognition.face_encodings(uploaded_image)

        if not known_encodings or not uploaded_encodings:
            logging.warning("Face not detected in one of the images")
            return jsonify({"match": False, "message": "Face not detected in one of the images"}), 400

        result = face_recognition.compare_faces([known_encodings[0]], uploaded_encodings[0])
        match_result = bool(result[0])  # convert numpy.bool_ to native bool
        logging.info(f"Face match result: {match_result}")

        return jsonify({
            "match": match_result,
            "message": "Match" if match_result else "No match"
        })

    except Exception as e:
        logging.error(f"Face recognition error: {e}")
        return jsonify({"match": False, "message": f"Face recognition failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
