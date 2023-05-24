from ggplot.scales.scale_color_gradient import colors_at_breaks
import pandas as pd

df = pd.read_excel("timeseries.xlsx")


# from dateutil import parser
# df["date"] = df.date.apply(parser.parse)

from snownlp import SnowNLP

def get_sentiment_cn(text):
    s = SnowNLP(text)
    return s.sentiments

df["sentiment"] = df.comment.apply(get_sentiment_cn)
print(df.head)

from ggplot import *
image = ggplot(aes(x="date", y="sentiment"), data=df) + geom_point() + geom_line(size = 5,color = 'darkblue') + scale_x_date(labels = date_format("%Y-%m-%d"))
print(image)

# import plotly as py
# from plotly.graph_objs import scatter, layout, Data

# trace = scatter(
#     x = [1,2,3,4],
#     y = [10,15,13,17]
# )

# data = Data([trace])    
# py.offline.plot(data, filename = "first_offline_start")

# import datetime

# x = datetime.datetime.strptime(df["date"][0],"%Y-%m-%d")
# print(x)