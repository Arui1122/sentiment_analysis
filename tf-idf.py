#encoding=utf-8
import jieba

jieba.set_dictionary("dict_text_big.txt")
jieba.load_userdict("user_dict.txt")

with open("comments.txt","r",encoding="utf-8") as file:
    sentence = file.read()

with open("stop_words.txt","r",encoding="utf-8") as file:
    stopwords = file.read().split("\n")


breakword = jieba.cut(sentence,cut_all=False)
final_words = []
for word in breakword:
    if word not in stopwords:
        final_words.append(word)
print( " ".join(final_words))

x = str(final_words)

import jieba.analyse
tags = jieba.analyse.extract_tags(x, topK=30, withWeight=True)

for tag in tags:
    print('word:', tag[0], 'tf-idf:', tag[1])