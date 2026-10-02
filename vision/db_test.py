import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="mysql123",
    database="parking"
)

cursor = db.cursor()

final_plate = "123가4568"

cursor.execute("SELECT * FROM cars WHERE plate_number = %s", (final_plate,))
car = cursor.fetchone()

if car:
    print("등록 차량:", car)
else:
    print("미등록 차량")

cursor.close()
db.close()
