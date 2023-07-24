# This is a sample Python script.

import pickle

# 读取pickle文件并反序列化
with open('char_dict.pkl', 'rb') as f:
    data = pickle.load(f)

# 打印反序列化后的数据
print(data)
