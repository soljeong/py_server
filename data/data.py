import csv
import mysql.connector

# MySQL 데이터베이스 연결
conn = mysql.connector.connect(
    host='127.0.0.1',
    port=3306,
    user='urstory',
    password='u1234',
    database='examplesdb',
    ssl_disabled=True
)
cursor = conn.cursor()

# 판매 내역 테이블 생성
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        sales_id BIGINT AUTO_INCREMENT PRIMARY KEY NOT NULL,
        date TEXT,
        product_id BIGINT,
        FOREIGN KEY (product_id) REFERENCES product(product_id),
        quantity INTEGER
    )
''')

# CSV 파일에서 데이터 읽어와서 데이터베이스에 삽입
with open('./data/train_v3.csv', 'r', encoding='utf-8') as csv_file:
    print("불러왔어")
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)  # 첫 번째 행은 헤더이므로 건너뜁니다.

    for row in csv_reader:
        date = row[0]
        for product_id, quantity in enumerate(row[1:], start=1):
            cursor.execute('''
                INSERT INTO sales (date, product_id, quantity)
                VALUES (%s, %s, %s)
            ''', (date, product_id, int(quantity)))
            print("입력완료")

# 변경사항 저장
conn.commit()

# 연결 종료
conn.close()
