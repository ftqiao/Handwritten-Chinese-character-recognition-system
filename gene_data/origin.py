import os
import pickle

# 常用汉字3755个
with open('char_dict.pkl', 'rb') as f:
    data = pickle.load(f)


def generate_key_value_data():
    image_dir = "F:\\captchaFiles\\distinctTest\\gen_words"
    myset = set()
    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        if os.path.isfile(file_path):
            cha = file_path.split("\\")[4][0]  # F:\\captchaFiles\\300\\gen_words\\吃-239-2.png
            if(cha=='d'):
                print(file_path)
            myset.add(cha)
    print(len(myset))
    my_lst = list(myset)
    # for item in myset:
    #     print(item)
    # 去掉在本地测试集但是不在3375个常用字的数据，一共有986个数据
    for key in my_lst:
        print("'" + key + "'", ':', data.get(key), ',')
    return my_lst


if __name__ == '__main__':
    test_lst = generate_key_value_data()
