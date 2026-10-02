from flask import Flask, request, jsonify
import mysql.connector

app = Flask(__name__)

# 서버: 이 경로에 POST 요청이 오면 아래 함수 실행 
@app.route("/check_plate", methods=["POST"])
def check_plate():
    data = request.get_json(silent=True)    # 전송받은 JSON 읽기

    if not isinstance(data, dict):
        return jsonify({"error": "JSON 객체가 필요합니다"}), 400

    plate = data.get("plate_number")

    if not isinstance(plate, str) or not plate.strip():
        return jsonify({"error": "번호판이 필요합니다"}), 400

    plate = plate.strip()

    db = mysql.connector.connect(
        host="localhost",
        user="parking_user",
        password="mysql123",
        database="parking"
    )

    cursor = db.cursor()

    try:
        cursor.execute(
            "SELECT 1 FROM cars WHERE plate_number = %s LIMIT 1",
            (plate,)
        )

        registered = cursor.fetchone() is not None

        print(f"번호판: {plate} / 등록 여부: {registered}")

        return jsonify({
            "plate_number": plate,
            "registered": registered
        })

    finally:
        cursor.close()
        db.close()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)