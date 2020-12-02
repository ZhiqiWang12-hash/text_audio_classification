def load_emotion_dict(path):
    work_book = xlrd.open_workbook(path)
    sheet_1 = work_book.sheet_by_index(0)
    rows = sheet_1.get_rows()
    data_row = []
    for row in range(sheet_1.nrows):
        data_row.append(sheet_1.row_values(row))
    return data_row

def org_dict(emotion_list,number,data_row):
    emotion_type8={}
    number_index=0
    for i in range(len(emotion_list)):
        if i>=number[number_index]:
            number_index+=1
        emotion_type8[emotion_list[i]]=number_index
    #print(emotion_type8)

    emotion_dict={}
    for row in data_row[1:]:
        key=row[0]
        value_list=[]
        #print(row[4]+'nononon')
        value_list.append(emotion_type8[re.sub(' ','',row[4])])
        value_list.append(row[5])
        value_list.append(row[6])
        emotion_dict[key]=value_list
    print(len(emotion_dict))
    return emotion_dict

def get_emotion_feature(X,emotion_dict):
    X_emo_feature=np.zeros((len(X),11))
    for i in range(len(X)):
        for j in range(len(X[i])):
            word = X[i][j]
            if word in emotion_dict.keys():
                dic_list=emotion_dict[word]
                index=emotion_dict[word][0]
                level=emotion_dict[word][1]
                polarity=int(emotion_dict[word][2]+7)
            #print(polarity)
                X_emo_feature[i,index]+=level
                X_emo_feature[i,polarity]+=1
    pca = PCA(n_components=11)
    newX = pca.fit_transform(X_emo_feature)
    return newX.reshape(-1,11,1)
def emo_dict_feature(data_row,emotion_list,number,X):
    data_row=load_emotion_dict(emotion_path)
    emotion_dict=org_dict(emotion_list,number,data_row)
    X_emo_f=get_emotion_feature(X,emotion_dict)
    return X_emo_f


emotion_path='./大连理工大学情感词汇本体库.xlsx'
data_row=load_emotion_dict(emotion_path)
emotion_list=['PA','PE', #happiness
              'PD','PH','PG','PB','PK', #like
              'NA',#anger
              'NB','NJ','NH','PF',#sadness
              'NI','NC','NG',#fear
              'NE','ND','NN','NK','NL',#hatress
              'PC']#surprise
number=[2,7,8,12,15,20,21]
