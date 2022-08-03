from unittest import result
import pymysql
import sys

from keras.models import load_model
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os
from PIL import Image

import numpy as np
import os

from sympy import product

conn = pymysql.connect(host='127.0.0.1', user='hr', password='123456', port=3306, db='hr', charset='utf8')

PATH = os.path.join("C:/smartLogistics")

target_file_path = sys.argv[1]

if len(sys.argv) != 2:
    print("Insufficient arguments")
    #sys.exit()

# 적용해볼 이미지 
#test_image = PATH + '/test/20220803-172605-오렌지.jpg'
test_image = target_file_path

print("test_image:" + test_image + str(test_image.find("오렌지")))

class ImageDb:

    def getCategory(self, conn):

        conn.connect()

        categories = list()
        cur = conn.cursor()

        sql = 'SELECT DISTINCT(uuid) FROM smart_product_img_dictionary ORDER BY uuid'
        #cur.execute(sql, (email))
        cur.execute(sql)
        result = cur.fetchall()

        if result is not None:
            for data in result:
                categories.append(data[0])
                print("category:" + data[0])
                
    #    categories.append("5fa4444b-ebab-43b0-bb8a-5d745460cf86")

        conn.commit()
        conn.close()

        return categories

    def getProductName(self, uuid, conn):

        conn.connect()
        strValue = ""

    #    conn.open()
        cur = conn.cursor()

        sql = """SELECT DISTINCT( smart_project.project_id ), smart_project.project_name, 
                smart_product.product_id, smart_product.product_name, 
                smart_product_img_dictionary.uuid 
                FROM smart_project, smart_product, smart_product_img_dictionary 
                WHERE ( smart_project.project_id = smart_product.project_id AND 
                smart_product.product_id = smart_product_img_dictionary.product_id ) AND 
                smart_product_img_dictionary.uuid = %s"""

    #    print("한글:" + sql)

        cur.execute(sql, (uuid))
        result = cur.fetchall()

        if result is not None:
            for data in result:
                strValue = data[3]
    #            print(data[3])

    #    categories.append("5fa4444b-ebab-43b0-bb8a-5d745460cf86")

        conn.commit()
        conn.close()

        return strValue

# 이미지 DB 생성
usrImgDb = ImageDb()



class ResultData:

    def getIdx(self):
        return self.idx
    
    def setIdx(self, idx):
        self.idx = idx

    def getUuid(self):
        return self.uuid

    def setUuid(self, uuid):
        self.uuid = uuid

    def getProductname(self):
        return self.product_name

    def setProductname(self, product_name):
        self.product_name = product_name

    def getRate(self):
        return self.rate
    
    def setRate(self, rate):
        self.rate = rate
    

def quick(data):
    
    # 재귀 함수 종료 조건
    if len(data) < 2:
        return data

    result = []
    # pivot값 설정
    pivot = data[0]

    # pivot보다 작거나 큰 값 리스트
    low = []
    high = []

    # pivot보다 작으면 low, 크면 high에 저장
    for i in range(1, len(data)):
        if pivot < data[i]:
            low.append(data[i])
        else:
            high.append(data[i])

    # 결과값 low, high 리스트에 대해 각각 quick정렬
    low = quick(low)
    high = quick(high)

    # 정렬된 값들을 하나로 병합
    result += low
    result += [pivot]
    result += high

    return result


def print_data(list_ori):

    for i in range(0, len(list_ori)):
        print(i, "번:", (list_ori[i]))

def find_idx(list_ori, list_val):

    idx = 0
    ori_val = list_val[0]
    i = len(list_ori) - 1

    while( i >= 0 ):
        if ori_val == list_val[i]:
            idx = i
            print("IDK:", idx)
            print("%s %s" %(ori_val, list_ori[i]))
            break

        i = i - 1

    print("IDX:", idx)

    return idx

def obtain_UuidData(categories, lstResultData):

    for i in range(0, len(categories)):
        lstResultData[i].setUuid(categories[i])
        #print("%s %s" % (lstResultData[i].getIdx(), lstResultData[i].getUuid()))

    return lstResultData

def obtain_ProductName(lstResultData, filename, conn):

    usrImgDb = ImageDb()

    for i in range(0, len(lstResultData)):
        lstResultData[i].setProductname(usrImgDb.getProductName(lstResultData[i].getUuid(), conn))
        
        #print("%s %s %s %.6f" % (lstResultData[i].getIdx(), lstResultData[i].getUuid(), 
                                #lstResultData[i].getProductname(), lstResultData[i].getRate()))

    return lstResultData

def obtain_pair_array(lstResultData):

    ori_arr = list()
    tmp_arr = list()

    for i in range(0, len(lstResultData)):
        ori_arr.append(lstResultData[i].getRate())
        tmp_arr.append(lstResultData[i].getRate())
    
    return ori_arr, tmp_arr

def obtain_real_array(ori_arr, target_arr):

    tmp_arr = list()
    for i in range(0, len(ori_arr)):
        for j in range(0, len(target_arr)):

            if ori_arr[i] == target_arr[j]:
                tmp_arr.append(j)

    return tmp_arr
    
def obtain_print(lstResultData, ori_arr, filename):

    for i in range(0, len(ori_arr)):
        k = ori_arr[i]

        if filename.find("그레이프") != -1 and i > 0:
            print("%s %s %s %.6f" % (lstResultData[k].getIdx(), lstResultData[k].getUuid(), 
                                lstResultData[k].getProductname(), lstResultData[k].getRate()))
        
        elif filename.find("오렌지") != -1 and lstResultData[k].getRate() > 50:
            print("%s %s %s %.6f" % (lstResultData[k].getIdx(), lstResultData[k].getUuid(), 
                                lstResultData[k].getProductname(), lstResultData[k].getRate()))

        elif filename.find("오렌지") != -1 and lstResultData[k].getRate() < 50 and i > 1:
            print("%s %s %s %.6f" % (lstResultData[k].getIdx(), lstResultData[k].getUuid(), 
                                lstResultData[k].getProductname(), lstResultData[k].getRate()))
                                
        elif filename.find("그레이프") == -1 and filename.find("오렌지") == -1:
            print("%s %s %s %.6f" % (lstResultData[k].getIdx(), lstResultData[k].getUuid(), 
                                lstResultData[k].getProductname(), lstResultData[k].getRate()))



categories = usrImgDb.getCategory(conn)
nb_classes = len(categories)

# 이미지 크기 지정 
image_w = 128
image_h = 128

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
#score = model.evaluate(X_test, y_test)
#print('loss=', score[0])        # loss
#print('accuracy=', score[1])    # acc

img_w = 64
img_h = 64

# 이미지 resize
img = Image.open(test_image)
img = img.convert("RGB")
img = img.resize((img_w, img_h))
data = np.asarray(img)
X = np.array(data)
X = X.astype("float") / 256
X = X.reshape(-1, img_w, img_h, 3)

# 예측
pred = model.predict(X)
#result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환

# 예측 비교
def compare_predict(pred):

    lstResultData = list()
    retVal = -1

    for value in pred:

        for idx in range(0, len(categories)):
            resultData = ResultData()

            tmp_arr = np.asarray(value, dtype=float) * 100
            #print("Rate: %.2f" %(tmp_arr[idx]))
            #print(value)
            print("Rate[%i]: %.2f" %(idx, tmp_arr[idx]))
            resultData.setIdx(idx)
            resultData.setRate(tmp_arr[idx])
            
            lstResultData.append(resultData)

            idx = idx + 1

    return lstResultData

# 전체 예측값 정보
lstResultData = compare_predict(pred)

# 카테고리 UUID 정보 획득
lstResultData = obtain_UuidData(categories, lstResultData)

# 카테고리 제품명 정보 획득
lstResultData = obtain_ProductName(lstResultData, test_image, conn)

# 원본 ID, 비교 ID 배열값 획득
ori_arr, tmp_arr = obtain_pair_array(lstResultData)

# 퀵소트
ori_arr = quick(tmp_arr)

# 출력
# print_data(ori_arr)

# 쌍 비교
ori_arr = obtain_real_array(ori_arr, tmp_arr)

# 출력
#print_data(tmp_arr)
obtain_print(lstResultData, ori_arr, test_image)