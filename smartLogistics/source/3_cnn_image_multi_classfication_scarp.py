import pymysql
from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

conn = pymysql.connect(host='100.30.0.5', user='hr', password='123456', port=3306, db='hr', charset='utf8')

# 1단계 - 데이터 수집
# 분류 대상 카테고리 선택하기 
def getCategory(conn):

    categories = list()
    cur = conn.cursor()

    sql = 'SELECT DISTINCT(uuid) FROM smart_product_img_dictionary ORDER BY uuid'
    #cur.execute(sql, (email))
    cur.execute(sql)
    result = cur.fetchall()

    if result is not None:
        
        for data in result:
            categories.append(data[0])
            print(data[0])

#    categories.append("5fa4444b-ebab-43b0-bb8a-5d745460cf86")

    conn.commit()
    conn.close()

    return categories


PATH = os.path.join("C:/smartLogistics")

categories = getCategory(conn)
nb_classes = len(categories)

# 이미지 크기 지정 
image_w = 64
image_h = 64
pixels = image_w * image_h * 3

# 이미지 데이터 읽어 들이기 
X = []
Y = []

for idx, cat in enumerate(categories):
    # 레이블 지정 
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 
    image_dir = PATH + "/train/" + cat
    files = glob.glob(image_dir + "/*.jpg")

    for i, f in enumerate(files):
        img = Image.open(f) 
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)      # numpy 배열로 변환
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)

X = np.array(X)
Y = np.array(Y)

# 학습 전용 데이터와 테스트 전용 데이터 구분 
X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)

print('>>> data 저장중 ...')
np.save( PATH + "/7obj.npy", xy)
print("ok,", len(Y))