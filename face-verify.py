from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import os
import logging
from PIL import Image
import io
import numpy as np
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Path ke folder gambar user
BASE_USER_IMAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'boldeaccess-backend', 'assets', 'users')
)

# Toleransi face distance
TOLERANCE = 0.65

def resize_image(image_array, max_width=800):
    height, width = image_array.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image_array, new_size)
    return image_array

@app.route("/")
def health_check():
    return "Face Recognition API is running!", 200

@app.route("/face-verify", methods=["POST"])
def verify_face():
    filename = request.form.get("user_image_name")
    uploaded_file = request.files.get("image")
    # DEBUG LOGGING
    logging.info(f"Form data - user_image_name: {filename}")
    logging.info(f"Form data - uploaded_file: {uploaded_file.filename if uploaded_file else 'None'}")

    if not uploaded_file or not filename:
        logging.warning("Permintaan tidak valid: gambar atau nama file kosong")
        return jsonify({"match": False, "message": "Gambar dan nama file wajib dikirim"}), 400

    # Baca dan konversi gambar upload ke format JPEG
    try:
        img = Image.open(uploaded_file)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        uploaded_image = face_recognition.load_image_file(buffer)
        uploaded_image = resize_image(uploaded_image)
    except Exception as e:
        logging.error(f"Gagal membaca gambar upload: {e}")
        return jsonify({"match": False, "message": f"Gagal membaca gambar upload: {str(e)}"}), 500

    # Path gambar user
    known_face_path = os.path.join(BASE_USER_IMAGE_DIR, filename)
    if not os.path.exists(known_face_path):
        logging.error(f"Gambar user tidak ditemukan: {filename}")
        return jsonify({"match": False, "message": "Gambar user tidak ditemukan"}), 404

    # Coba ambil cache encoding
    encoding_cache_path = known_face_path + ".npy"
    try:
        if os.path.exists(encoding_cache_path):
            known_encoding = np.load(encoding_cache_path)
        else:
            known_img = Image.open(known_face_path)
            if known_img.mode in ("RGBA", "P"):
                known_img = known_img.convert("RGB")
            buffer_known = io.BytesIO()
            known_img.save(buffer_known, format="JPEG")
            buffer_known.seek(0)
            known_image = face_recognition.load_image_file(buffer_known)
            known_image = resize_image(known_image)

            known_encodings = face_recognition.face_encodings(known_image)
            if not known_encodings:
                return jsonify({"match": False, "message": "Wajah tidak terdeteksi di gambar user"}), 400
            known_encoding = known_encodings[0]
            np.save(encoding_cache_path, known_encoding)
    except Exception as e:
        logging.error(f"Gagal memproses gambar user: {e}")
        return jsonify({"match": False, "message": f"Gagal memproses gambar user: {str(e)}"}), 500

    # Ambil encoding gambar upload
    try:
        uploaded_encodings = face_recognition.face_encodings(uploaded_image)
        if not uploaded_encodings:
            return jsonify({"match": False, "message": "Wajah tidak terdeteksi di gambar upload"}), 400
        uploaded_encoding = uploaded_encodings[0]

        distance = face_recognition.face_distance([known_encoding], uploaded_encoding)[0]
        is_match = distance < TOLERANCE

        logging.info(f"Jarak wajah: {distance:.4f} | Toleransi: {TOLERANCE} | Cocok: {is_match}")
        return jsonify({
            "match": bool(is_match),
            "distance": float(distance),
            "tolerance": TOLERANCE,
            "message": "Cocok" if is_match else "Tidak cocok"
        }), 200
    except Exception as e:
        logging.exception("Kesalahan saat melakukan verifikasi wajah")
        return jsonify({"match": False, "message": f"Face recognition gagal: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
