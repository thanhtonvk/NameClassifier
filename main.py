import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
import xgboost

'''path_data_train: là đường dẫn tới file csv train cho mô hình vd: data_train_moredata6.csv'''
path_data_train = 'data_train_moredata6.csv'

'''load model xgboost'''
model = xgboost.XGBClassifier()
model.load_model('models/bestmodel.bin')


'''
real_x: dữ liệu để dự đoán
'''
def preprocessing(real_x):
    
    data_train = pd.read_csv(path_data_train)     
    x_train = data_train["name"]
    
    # ngram-char level - we choose max number of words equal to 30000 except all words (100k+ words)
    tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', max_features=30000, ngram_range=(2, 3))
    tfidf_vect_ngram_char.fit(x_train)
    # assume that we don't have test set before
    real_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(real_x)
    return real_tfidf_ngram_char

'''
real_x: dữ liệu để dự đoán
file_path_export: đường dẫn xuất file kết quả
'''
def predict_to_csv(id,real_x,file_path_export):
    real_tfidf_ngram_char = preprocessing(real_x)
    real_predictions = model.predict(real_tfidf_ngram_char)
    real_predictions_proba = model.predict_proba(real_tfidf_ngram_char)
    results = pd.DataFrame([id,real_x,  real_predictions, real_predictions_proba[:,0]]).transpose()
    results.columns = ["id","name",  "ket qua du doan", "kha nang non_person"]
    results.to_csv(file_path_export)
    return file_path_export

def read_file(path):
    f = open(path,'r',encoding = 'utf-8')
    lines = f.readlines()
    total_file = int(len(lines)/1000000 + 1)
    for i in range(total_file):
        if i<total_file-1:
            data = lines[i*1000000:(i+1)*1000000]
        else:
            data = lines[(i-1)*1000000:]
        save_file =open(f"export/{i}.csv",'a',encoding='utf-8')
        save_file.writelines(data)
        save_file.close()
        break
    f.close()

import os
def predict_all():
    ROOT_DIR = 'export/'
    files = os.listdir(ROOT_DIR)
    for file in files:
        path = ROOT_DIR+file
        print(path)
        real_data = pd.read_csv(path,header=None, names=['id','name'])
        full_name =[]
        id = []
        for i in real_data.index:
            full_name.append(unidecode(real_data["name"][i]).lower())
            id.append(real_data["id"][i])
        real_data = pd.DataFrame([id,full_name])
        real_data = real_data.transpose()
        real_data.columns = ["id","name"]
        real_x = real_data["name"]
        file_path_export = f"predict/{file}"
        predict_to_csv(id,real_x,file_path_export)
        



if __name__=="__main__":
    # đọc và tách file
    read_file('data_test.csv')
    
    predict_all()
    
    
    
    


 
    
    