import mysql.connector
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from lightgbm import LGBMRegressor
from model_handler import ModelHandler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from decimal import Decimal


# 파일 불러오기
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv("./data/sample_submission.csv")
# '일시' 컬럼을 datetime 형식으로 변환하고, Feature을 추출
train_df['일시'] = pd.to_datetime(train_df['일시'])
train_df['년'] = train_df['일시'].dt.year
train_df['월'] = train_df['일시'].dt.month
train_df['일'] = train_df['일시'].dt.day
test_df['일시'] = pd.to_datetime(test_df['일시'])
test_df['년'] = test_df['일시'].dt.year
test_df['월'] = test_df['일시'].dt.month
test_df['일'] = test_df['일시'].dt.day
test_feature = test_df[['년','월','일']]
train_features = train_df[test_feature.columns]
train_target = train_df.drop(columns=['일시', '년','월','일']).copy()
target_columns = train_target.columns
# 교차 검증
kf = KFold(n_splits=2, shuffle=True, random_state=42)
scoring = 'neg_mean_absolute_error' # 어떤 평가지표를 사용할 것인가?
# 모델을 저장할 딕셔너리 생성
models_dict = {}
best_params = {
    "max_depth": 8,
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "random_state": 42,
    "verbose": -1
}
# 모델 핸들러 인스턴스 생성
model_handler = ModelHandler()
# 학습된 모델을 모델 핸들러에 전달
model_handler.set_models(models_dict)
# 각 목표 변수에 대한 모델 학습
for target_col in target_columns:
    # best 모델 인스턴스 생성(정의)
    lgb_model_target = LGBMRegressor(**best_params)
    n_iter = 0
    accuracy_lst = []
    mae_lst = []
    for train_index, valid_index in kf.split(train_features, train_target[target_col]):
        n_iter += 1
        # 학습용, 검증용 데이터 구성
        train_x, valid_x = train_features.iloc[train_index], train_features.iloc[valid_index]
        train_y, valid_y = train_target[target_col].iloc[train_index], train_target[target_col].iloc[valid_index]
        # 학습
        lgb_model_target.fit(train_x, train_y)
        valid_pred = lgb_model_target.predict(valid_x) # 예측값
        # 평가
        mae = mean_absolute_error(valid_y, valid_pred)
        mae_lst.append(mae)
        print(f'{n_iter} 번째 Stratified K-Fold MAE for {target_col}: {mae}')
    # 학습된 모델 저장
    model_handler.train_model(target_col, lgb_model_target)





def get_db():
    # 데이터베이스에 연결
    conn = mysql.connector.connect(
        host='127.0.0.1',
        port=3306,
        user='urstory',
        password='u1234',
        database='examplesdb',
        ssl_disabled=True
    )
    return conn

# FastAPI 앱 생성
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# 홈 페이지
@app.get("/admin")
def read_root(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/api/sales-data")
def get_sales_data():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    # SQL 쿼리 실행
    sql_query = '''
    SELECT
        DATE_FORMAT(s.date, '%Y-%m') AS month,
        SUM(s.quantity * p.product_price) AS total_sales
    FROM
        sales s
    JOIN
        product p ON s.product_id = p.product_id
    GROUP BY
        DATE_FORMAT(s.date, '%Y-%m')
    ORDER BY
        month;
    '''
    cursor.execute(sql_query)
    
    # Convert Decimal objects to float for JSON serialization
    sales_data_serializable = [
        {"month": row["month"], "total_sales": float(row["total_sales"])} 
        for row in cursor.fetchall()
    ]

    # 연결 종료
    cursor.close()
    conn.close()
    
    return JSONResponse(content=sales_data_serializable)




@app.get("/api/day5-data")
def get_day5_data():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    # SQL 쿼리 실행
    sql_query = '''
    SELECT
        DATE_FORMAT(s.date, '%Y-%m-%d') AS day,
        SUM(s.quantity * p.product_price) AS total_sales
    FROM
        sales s
    JOIN
        product p ON s.product_id = p.product_id
    GROUP BY
        DATE_FORMAT(s.date, '%Y-%m-%d')
    ORDER BY
        day DESC
    LIMIT 5;
    '''
    cursor.execute(sql_query)
    
    # Convert Decimal objects to float for JSON serialization
    day5_data_serializable = [
        {"day": str(row["day"]), "total_sales": float(row["total_sales"])} 
        for row in cursor.fetchall()
    ]

    # 연결 종료
    cursor.close()
    conn.close()
    
    return JSONResponse(content=day5_data_serializable)

@app.get("/api/products")
def read_products():
    # 데이터베이스에 연결
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    # SQL 쿼리 실행
    sql_query = '''
        SELECT
            p.product_id,
            p.product_name,
            p.product_price,
            p.product_stock,
            SUM(s.quantity) as total_quantity,
            DATE_FORMAT(s.date, '%Y-%m-%d') AS day
        FROM
            sales s
        JOIN
            product p ON s.product_id = p.product_id
        GROUP BY
            p.product_id, p.product_name, p.product_price, p.product_stock, day
        ORDER BY
            day DESC
        LIMIT 52;
    '''
    cursor.execute(sql_query)
    # 예측 결과
    predictions = {target_col: int(model_handler.predict(target_col, pd.DataFrame({'년': [2023], '월': [1], '일': [1]}))[0]) for target_col in models_dict.keys()}
    # cursor로 저장
    products = []
    for row in cursor:
        product_id = row["product_id"]
        product_name = row["product_name"]
        product_price = row["product_price"]
        product_stock = row["product_stock"]
        total_quantity = row["total_quantity"]

        # daily_sales_prediction 계산
        daily_sales_prediction = predictions.get(str(product_id), 0)

        # 필요한 주문량 계산
        order_quantity = max(0, int(predictions.get(str(product_id), 0)) - product_stock)

        # 제품 정보를 products 리스트에 추가
        products.append({
            "product_name": product_name,
            "product_price": int(product_price),
            "product_stock": product_stock,
            "total_quantity": total_quantity,
            "daily_sales_prediction": daily_sales_prediction,
            "order_quantity": order_quantity
        })

    # 연결 종료
    cursor.close()
    conn.close()
    return products