# 先训练一次，再在第一次训练得到的模型的基础上进行训练七次，当有某一次测试准确率达到0.95以上的时候停止训练

import get_model
import get_train_array
import get_test_array
import time
from keras import models
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks

model = get_model.get_model()  # 选择模型

# 加载训练数据和测试数据
# (train_image, val_image, train_label, val_label) = get_train_array.load_data('F:/handwrite/recognize_system/train/')
# (test_image, test_label) = get_test_array.load_data('F:/handwrite/recognize_system/test/')\
(train_image, val_image, train_label, val_label) = get_train_array.load_data('data/train/')
(test_image, test_label) = get_test_array.load_data('data/test/')

logs_path = "./logs/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1)
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=20, verbose=1)
model.fit(train_image, train_label, validation_data=(val_image, val_label),
          epochs=100, batch_size=128,
          callbacks=[tensorboard_callback, early_stopping])
# 训练, fit方法自带shuffle随机读取
# model.fit(train_image, train_label, validation_data=(val_image, val_label))
test_scores = model.evaluate(test_image, test_label)
test_accuracy = test_scores[1]

# 将模型保存为 HDF5 文件
strTime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
print('Time:', strTime, '   ', 'Test Accuracy:', test_accuracy)
model.save('Chinese_recognition_model_v3.h5')

# for i in range(1, 8):
#     print()
#     model = models.load_model('Chinese_recognition_model_v3.h5')
#     model.fit(train_image, train_label)
#     test_scores = model.evaluate(test_image, test_label)
#     test_accuracy = test_scores[1]
#     strTime=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
#     print('Time:', strTime, '   ', 'Test Accuracy:', test_accuracy)
#     model.save('Chinese_recognition_model_v3.h5')
#     if test_accuracy > 0.95:
#         break
