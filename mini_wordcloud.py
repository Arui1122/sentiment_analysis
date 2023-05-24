from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import numpy as np
from collections import Counter

text = open("mini_comment","r",encoding="utf-8").read()

with open("stop_words.txt","r",encoding="utf-8") as f:
    stops = f.read().split("\n")
terms = []


