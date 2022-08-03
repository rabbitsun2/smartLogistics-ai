import pymysql

from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

# 2단계 - 데이터 모델 생성
# 분류 대상 카테고리 선택하기 
conn = pymysql.connect(host='100.30.0.5', user='hr', password='123456', port=3306, db='hr', charset='utf8')

PATH = os.path.join("C:/smartLogistics")

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


categories = getCategory(conn)

nb_classes = len(categories)

# 이미지 크기 지정 
image_w = 64
image_h = 64

# 데이터 열기 
X_train, X_test, y_train, y_test = np.load(PATH + "/7obj.npy", allow_pickle=True)

# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)

# 모델 구조 정의 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(512))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])

# 모델 확인
#print(model.summary())

# 학습 완료된 모델 저장
hdf5_file = PATH + "/7obj-model.hdf5"

if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    model.fit(X_train, y_train, epochs=30, batch_size=32)
    model.save_weights(hdf5_file)

# 모델 평가하기 
score = model.evaluate(X_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc