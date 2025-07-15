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

# Path ke folder gambar user (ubah sesuai struktur project-mu)
BASE_USER_IMAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'boldeaccess-backend', 'assets', 'users')
)

# Toleransi threshold face-recognition
TOLERANCE = 0.6

@app.route("/verify-face", methods=["POST"])
def verify_face():
    filename = request.form.get("user_image_name")
    if 'image' not in request.files or not filename:
        logging.error("Gambar atau nama file tidak ditemukan dalam request")
        return jsonify({"match": False, "message": "Gambar atau nama file tidak ditemukan"}), 400

    uploaded_file = request.files['image']
    logging.info(f"Request verifikasi wajah untuk: {filename}")

    try:
        uploaded_image = face_recognition.load_image_file(uploaded_file)
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
        known_image = face_recognition.load_image_file(known_face_path)
        logging.info("Gambar wajah user berhasil dibaca")
    except Exception as e:
        logging.error(f"Gagal membaca gambar user: {e}")
        return jsonify({"match": False, "message": f"Gagal membaca gambar user: {str(e)}"}), 500

    try:
        known_encodings = face_recognition.face_encodings(known_image)
        uploaded_encodings = face_recognition.face_encodings(uploaded_image)

        if not known_encodings or not uploaded_encodings:
            logging.warning("Wajah tidak terdeteksi pada salah satu gambar")
            return jsonify({"match": False, "message": "Wajah tidak terdeteksi pada gambar"}), 400

        # Menghitung distance dan memeriksa apakah di bawah threshold
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
