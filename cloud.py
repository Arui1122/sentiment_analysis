#encoding=utf-8
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import numpy as np
from collections import Counter

with open("comments.txt","r",encoding="utf-8") as file:
    text = file.read()

jieba.set_dictionary("dict_text_big.txt")
with open("stop_words.txt","r",encoding="utf-8") as f:
    stops = f.read().split("\n")

terms = []
for t in jieba.cut(text,cut_all=False):
    if t not in stops:
        terms.append(t)
diction = Counter(terms)
font = "C:\\Windows\\Fonts\\simsun.ttc"

wordcloud = WordCloud(background_color = "white",font_path=font)
wordcloud.generate_from_frequencies(diction)

plt.figure(figsize=(6,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# wordcloud.to_file("comments_wordcloud.png")