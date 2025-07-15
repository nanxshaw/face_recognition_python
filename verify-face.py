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
TOLERANCE = 0.6  # Kamu bisa ubah ke 0.65 atau 0.7

@app.route("/verify-face", methods=["POST"])
def verify_face():
    filename = request.form.get("user_image_name")
    if 'image' not in request.files or not filename:
        logging.error("Image file atau nama file user tidak dikirim.")
        return jsonify({"match": False, "message": "Image atau filename missing"}), 400

    uploaded_file = request.files['image']
    logging.info(f"Request diterima untuk user image: {filename}")

    try:
        uploaded_image = face_recognition.load_image_file(uploaded_file)
        logging.info("Gambar upload berhasil dimuat.")
    except Exception as e:
        logging.error(f"Gagal memuat gambar upload: {e}")
        return jsonify({"match": False, "message": f"Gagal memuat gambar upload: {str(e)}"}), 500

    known_face_path = os.path.join(BASE_USER_IMAGE_DIR, filename)
    logging.info(f"Membandingkan dengan gambar di path: {known_face_path}")

    if not os.path.exists(known_face_path):
        logging.error("Gambar user tidak ditemukan.")
        return jsonify({"match": False, "message": "Gambar user tidak ditemukan"}), 404

    try:
        known_image = face_recognition.load_image_file(known_face_path)
        logging.info("Gambar user (known image) berhasil dimuat.")
    except Exception as e:
        logging.error(f"Gagal memuat gambar user: {e}")
        return jsonify({"match": False, "message": f"Gagal memuat gambar user: {str(e)}"}), 500

    try:
        known_encodings = face_recognition.face_encodings(known_image)
        uploaded_encodings = face_recognition.face_encodings(uploaded_image)

        logging.info(f"Wajah terdeteksi di gambar user: {len(known_encodings)}")
        logging.info(f"Wajah terdeteksi di gambar upload: {len(uploaded_encodings)}")

        if not known_encodings or not uploaded_encodings:
            logging.warning("Wajah tidak terdeteksi di salah satu gambar.")
            return jsonify({"match": False, "message": "Wajah tidak terdeteksi di salah satu gambar"}), 400

        distance = face_recognition.face_distance([known_encodings[0]], uploaded_encodings[0])[0]
        is_match = distance <= TOLERANCE

        logging.info(f"Distance wajah: {distance:.4f}")
        logging.info(f"Tolerance digunakan: {TOLERANCE}")
        logging.info(f"Hasil pencocokan wajah: {'MATCH' if is_match else 'TIDAK MATCH'}")

        return jsonify({
            "match": is_match,
            "distance": distance,
            "tolerance": TOLERANCE,
            "message": "Match" if is_match else "No match"
        })

    except Exception as e:
        logging.error(f"Error saat face recognition: {e}")
        return jsonify({"match": False, "message": f"Face recognition gagal: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
