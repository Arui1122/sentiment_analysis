from codecs import encode
import snownlp
from snownlp import SnowNLP, sentiment
import pandas as pd

df = pd.read_excel("promax.xlsx")

def make_label(star):
    if star > 3:
        return 1
    else:
        return 0
    
df['sentiments_star'] = df.star.apply(make_label)

def snow_result(comment):
    s =SnowNLP(comment)
    if s.sentiments >=0.6:
        return 1
    else:
        return 0

df['snlp_sentiment_result']= df.comment.apply(snow_result)

for i in range(38):
    text = df.comment.iloc[i]
    def get_sentiment_cn(text):
        s = SnowNLP(text)
        return s.sentiments
df["sentiment"] = df.comment.apply(get_sentiment_cn)    #算出來給下面train_test用的

print(df.head(10))

df.to_excel('sentiments_promax2.xlsx')    #另存excel檔案

counts =0

for i in range(len(df)):
    if df.iloc[i,2]== df.iloc[i,3]:
        counts+=1

print(counts/len(df))   #利用第三方的套件SnowNLP，精確度只來到0.57

#---------------------------------------------------------------------------------------------#

import jieba

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

df['cut_comment'] = df.comment.apply(chinese_word_cut)
# print(df.head(5))
df.to_excel('sentiments_promax2.xlsx')

X = df['cut_comment']   #要劃分的樣本"特徵集"
y = df.sentiments_star  #劃分的樣本"結果"label 1,0

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3) #分割訓練集、測試集

from sklearn.feature_extraction.text import CountVectorizer  #計算單字在文件出現的次數

def get_custom_stopwords(stop_words_file):      #自定義stopwords_file
    with open(stop_words_file,encoding="utf-8") as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

stop_words_file = 'stop_words.txt'
stopwords = get_custom_stopwords(stop_words_file)

vect = CountVectorizer(max_df = 0.8,  #去除掉超過這一比例的文檔中出現的關鍵詞（過於平凡)
                       min_df = 3,    #去除掉低於這一數量的文檔中出現的關鍵詞（過於獨特）
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                       stop_words=frozenset(stopwords))    #設置停用詞表，這樣的詞我們就不會統計出來

#   為結合進一步之詞雲圖、時間序列及情感分析探討，將以 Scikit-Learn進行 TF 之計算與統計，決定關鍵詞

test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names()) #轉成矩陣看看有哪些特徵 feature_names 欄位名稱
print(test)

test.to_excel('TF_promax2.xlsx')

# from sklearn.feature_extraction.text import TfidfVectorizer
# vec = TfidfVectorizer()
# X = vec.fit_transform(test)
# test = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
# print(test)

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
X_train_vect = vect.fit_transform(X_train)
nb.fit(X_train_vect, y_train)   #fit擬合：原義指的是安裝、使適合的意思，它並不是一個訓練的過程，而是一個適配的過程，過程都是定死的，最後只是得到了一個統一的轉換的規則模型。
train_score = nb.score(X_train_vect, y_train)
print(train_score)  #訓練模型，用的是簡單貝氏分類器，用來和snowNLP相比

X_test_vect = vect.transform(X_test)
print(nb.score(X_test_vect, y_test))    #測試數據驗證精確度

from sklearn.metrics import classification_report, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
# NB_classifier = MultinomialNB()
# NB_classifier.fit(X_train_vect,y_train)
y_predict_test = nb.predict(X_test_vect)
cm = confusion_matrix(y_test,y_predict_test)
print(sns.heatmap(cm, annot = True))
print(classification_report(y_test,y_predict_test))

X_vec = vect.transform(X)
nb_result = nb.predict(X_vec)
df['nb_result'] = nb_result   #將結果放到data裡
# print(df.head())
df.to_excel("sentiments_promax2.xlsx")