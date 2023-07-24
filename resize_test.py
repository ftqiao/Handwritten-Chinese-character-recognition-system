import os
from PIL import Image

image_dir = 'D:\\yntrust\\PycharmProjects\\Handwritten-Chinese-character-recognition-system\\my_picture'
file_name = '万-614-3.png'
file_path = os.path.join(image_dir, file_name)
# file_path = 'D:\yntrust\PycharmProjects\Handwritten-Chinese-character-recognition-system\my_picture\七-5318-5.png'
with Image.open(file_path) as img:
    # 定义目标宽度
    # target_width = 28
    # # 计算目标高度，
    # target_height = 28
    # # 调整图片大小
    # target_size = (target_width, target_height)
    # resized_img = img.resize(target_size)
    # 将调整后的图片保存到文件
    real_path = os.path.join('D:\\yntrust\\PycharmProjects\\Handwritten-Chinese-character-recognition-system\\resize_shape', file_name)
    convert_img = img.convert('L')
    convert_img = convert_img.convert('RGB')
    convert_img.save(real_path)
