import get_model
import get_train_array
import get_test_array
import time
from keras import models
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks

model = get_model.get_model()  # 选择模型

# 加载训练数据和测试数据
(train_image, val_image, train_label, val_label) = get_train_array.load_data('F:/handwrite/recognize_system/train/')
(test_image, test_label) = get_test_array.load_data('F:/handwrite/recognize_system/test/')
# (train_image, val_image, train_label, val_label) = get_train_array.load_data('data/train/')
# (test_image, test_label) = get_test_array.load_data('data/test/')

logs_path = "./logs/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1)

# 使用 ModelCheckpoint 回调函数，保存测试准确率达到0.95以上的模型，并在训练到该模型时停止
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath='Chinese_recognition_model_v3.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1)

model.fit(train_image, train_label, validation_data=(val_image, val_label),
          epochs=50, batch_size=32,
          callbacks=[tensorboard_callback, checkpoint_callback])

# 加载测试准确率达到0.95以上的模型
model = models.load_model('Chinese_recognition_model_v3.h5')

# 在测试集上评估模型性能
test_scores = model.evaluate(test_image, test_label)
test_accuracy = test_scores[1]

# 输出测试准确率
strTime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
print('Time:', strTime, '   ', 'Test Accuracy:', test_accuracy)

# for i in range(1, 8):
#     print()
#     model.fit(train_image, train_label)
#     test_scores = model.evaluate(test_image, test_label)
#     test_accuracy = test_scores[1]
#     strTime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
#     print('Time:', strTime, '   ', 'Test Accuracy:', test_accuracy)
#     if test_accuracy > 0.95:
#         break
