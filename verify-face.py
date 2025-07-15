from flask import Flask, request, jsonify
import face_recognition
import os
import logging
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

# Path ke folder gambar user (ubah sesuai struktur project-mu)
BASE_USER_IMAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'boldeaccess-backend', 'assets', 'users')
)

# Toleransi threshold face-recognition
TOLERANCE = 0.8


@app.route("/")
def health_check():
    return "Face Recognition API is running!", 200


@app.route("/verify-face", methods=["POST"])
def verify_face():
    filename = request.form.get("user_image_name")
    if 'image' not in request.files or not filename:
        logging.error("Gambar atau nama file tidak ditemukan dalam request")
        return jsonify({"match": False, "message": "Gambar atau nama file tidak ditemukan"}), 400

    uploaded_file = request.files['image']
    logging.info(f"Request verifikasi wajah untuk: {filename}")

    # Konversi gambar upload ke JPEG di memory (menghindari crash dari PNG / RGBA)
    try:
        img = Image.open(uploaded_file)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        uploaded_image = face_recognition.load_image_file(buffer)
        logging.info("Gambar absensi berhasil dibaca")
    except Exception as e:
        logging.error(f"Gagal membaca gambar absensi: {e}")
        return jsonify({"match": False, "message": f"Gagal membaca gambar absensi: {str(e)}"}), 500

    known_face_path = os.path.join(BASE_USER_IMAGE_DIR, filename)
    logging.info(f"Mencari gambar wajah user di: {known_face_path}")

    if not os.path.exists(known_face_path):
        logging.error("Gambar wajah user tidak ditemukan")
        return jsonify({"match": False, "message": "Gambar wajah user tidak ditemukan"}), 404

    try:
        known_img = Image.open(known_face_path)
        if known_img.mode in ("RGBA", "P"):
            known_img = known_img.convert("RGB")
        buffer_known = io.BytesIO()
        known_img.save(buffer_known, format="JPEG")
        buffer_known.seek(0)
        known_image = face_recognition.load_image_file(buffer_known)
        logging.info("Gambar wajah user berhasil dibaca")
    except Exception as e:
        logging.error(f"Gagal membaca gambar user: {e}")
        return jsonify({"match": False, "message": f"Gagal membaca gambar user: {str(e)}"}), 500

    try:
        known_encodings = face_recognition.face_encodings(known_image)
        uploaded_encodings = face_recognition.face_encodings(uploaded_image)

        logging.info(f"Wajah terdeteksi → known: {len(known_encodings)} | uploaded: {len(uploaded_encodings)}")

        if not known_encodings:
            return jsonify({"match": False, "message": "Wajah tidak terdeteksi di gambar database"}), 400
        if not uploaded_encodings:
            return jsonify({"match": False, "message": "Wajah tidak terdeteksi di gambar yang diupload"}), 400

        # Hitung jarak wajah dan bandingkan
        distance = face_recognition.face_distance([known_encodings[0]], uploaded_encodings[0])[0]
        match_result = distance < TOLERANCE

        logging.info(f"Jarak wajah: {distance:.4f} | Toleransi: {TOLERANCE} | Cocok: {match_result}")

        return jsonify({
            "match": bool(match_result),
            "distance": float(distance),
            "tolerance": TOLERANCE,
            "message": "Cocok" if match_result else "Tidak cocok"
        })

    except Exception as e:
        logging.error(f"Error saat face recognition: {e}")
        return jsonify({"match": False, "message": f"Face recognition gagal: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
