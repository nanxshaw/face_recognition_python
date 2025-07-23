from flask import Flask, request, jsonify
import face_recognition
import os
import logging
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # CORS untuk semua route

# Logging konfigurasi
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

# Folder gambar user (ubah sesuai struktur project-mu)
BASE_USER_IMAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'boldeaccess-backend', 'assets', 'users')
)

# Toleransi face distance
TOLERANCE = 0.55

@app.route("/")
def health_check():
    return "Face Recognition API is running!", 200


@app.route("/verify-face", methods=["POST"])
def verify_face():
    filename = request.form.get("user_image_name")
    uploaded_file = request.files.get("image")

    if not uploaded_file or not filename:
        logging.warning("Permintaan tidak valid: gambar atau nama file kosong")
        return jsonify({
            "match": False,
            "message": "Gambar dan nama file wajib dikirim"
        }), 400

    # Baca gambar yang diupload
    try:
        img = Image.open(uploaded_file)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        uploaded_image = face_recognition.load_image_file(buffer)
    except Exception as e:
        logging.error(f"Gagal membaca gambar yang diupload: {e}")
        return jsonify({"match": False, "message": f"Gagal membaca gambar upload: {str(e)}"}), 500

    # Path gambar user
    known_face_path = os.path.join(BASE_USER_IMAGE_DIR, filename)
    if not os.path.exists(known_face_path):
        logging.error(f"Gambar user tidak ditemukan: {filename}")
        return jsonify({"match": False, "message": "Gambar user tidak ditemukan"}), 404

    # Baca gambar user
    try:
        known_img = Image.open(known_face_path)
        if known_img.mode in ("RGBA", "P"):
            known_img = known_img.convert("RGB")
        buffer_known = io.BytesIO()
        known_img.save(buffer_known, format="JPEG")
        buffer_known.seek(0)
        known_image = face_recognition.load_image_file(buffer_known)
    except Exception as e:
        logging.error(f"Gagal membaca gambar user: {e}")
        return jsonify({"match": False, "message": f"Gagal membaca gambar user: {str(e)}"}), 500

    try:
        known_encodings = face_recognition.face_encodings(known_image)
        uploaded_encodings = face_recognition.face_encodings(uploaded_image)

        logging.info(f"Deteksi wajah → user: {len(known_encodings)}, upload: {len(uploaded_encodings)}")

        if not known_encodings:
            return jsonify({"match": False, "message": "Wajah tidak terdeteksi di gambar user"}), 400
        if not uploaded_encodings:
            return jsonify({"match": False, "message": "Wajah tidak terdeteksi di gambar upload"}), 400

        distance = face_recognition.face_distance([known_encodings[0]], uploaded_encodings[0])[0]
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
    # Jalankan di semua IP agar bisa diakses dari HP via jaringan lokal
    app.run(host="0.0.0.0", port=5000)
