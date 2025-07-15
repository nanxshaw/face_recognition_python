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

# Lokasi gambar user
BASE_USER_IMAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'boldeaccess-backend', 'assets', 'users')
)

@app.route("/verify-face", methods=["POST"])
def verify_face():
    filename = request.form.get("user_image_name")
    if 'image' not in request.files or not filename:
        logging.error("Gambar atau nama file tidak ada dalam permintaan")
        return jsonify({"match": False, "message": "Gambar atau nama file tidak ditemukan"}), 400

    uploaded_file = request.files['image']
    logging.info(f"Menerima verifikasi wajah untuk file: {filename}")

    try:
        uploaded_image = face_recognition.load_image_file(uploaded_file)
        logging.info("Gambar absensi berhasil dimuat")
    except Exception as e:
        logging.error(f"Gagal memuat gambar absensi: {e}")
        return jsonify({"match": False, "message": f"Gagal memuat gambar absensi: {str(e)}"}), 500

    known_face_path = os.path.join(BASE_USER_IMAGE_DIR, filename)
    if not os.path.exists(known_face_path):
        logging.error("Gambar wajah pengguna tidak ditemukan")
        return jsonify({"match": False, "message": "Gambar pengguna tidak ditemukan"}), 404

    try:
        known_image = face_recognition.load_image_file(known_face_path)
        logging.info("Gambar wajah pengguna berhasil dimuat")
    except Exception as e:
        logging.error(f"Gagal memuat gambar pengguna: {e}")
        return jsonify({"match": False, "message": f"Gagal memuat gambar pengguna: {str(e)}"}), 500

    try:
        known_encodings = face_recognition.face_encodings(known_image)
        uploaded_encodings = face_recognition.face_encodings(uploaded_image)

        if not known_encodings or not uploaded_encodings:
            logging.warning("Wajah tidak terdeteksi dalam salah satu gambar")
            return jsonify({"match": False, "message": "Wajah tidak terdeteksi dalam salah satu gambar"}), 400

        result = face_recognition.compare_faces([known_encodings[0]], uploaded_encodings[0], tolerance=0.7)
        match_result = bool(result[0])
        logging.info(f"Hasil pencocokan wajah: {match_result}")

        return jsonify({
            "match": match_result,
            "message": "Wajah cocok" if match_result else "Wajah tidak cocok"
        })

    except Exception as e:
        logging.error(f"Error dalam proses face recognition: {e}")
        return jsonify({"match": False, "message": f"Proses verifikasi wajah gagal: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
