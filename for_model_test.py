"""from scipy import misc
import numpy as np
image = misc.imread('D:\\PycharmProjects\\Image_Enhance\\test_image\\original_images\\blackberry\\34.jpg')
I = np.asarray(image)
IMAGE_SIZE = 3120*4160*3
I = np.float16(np.reshape(I, [1, IMAGE_SIZE]))/255
print(type(I))
print(I.shape)"""
import numpy as np
import tensorflow as tf
from scipy import misc

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3
batch_size = 5
dslr_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])
adv_ = tf.placeholder(tf.float32, [None, 1])
dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image),[-1, PATCH_WIDTH * PATCH_HEIGHT])
#adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
adversarial_ = tf.multiply(dslr_gray, adv_)
adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])
discrim_target = tf.concat([adv_, 1 - adv_], 1)
with tf.Session() as sess:
    image = np.float16(misc.imread("test_image\\patches\\iphone\\iphone\\1.jpg")) / 255
    image_ = np.reshape(image, [1, PATCH_SIZE])
    all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])

    print("discrim_target")
    print(sess.run(discrim_target, feed_dict={dslr_: image_, adv_:all_zeros}))
    print(discrim_target)
    print("dslr_gray")
    print(sess.run(dslr_gray, feed_dict={dslr_: image_, adv_:all_zeros}))
    print(dslr_gray)
    print('image_')
    print(image_.shape)
    print('all_zeros')
    print(all_zeros.shape)
