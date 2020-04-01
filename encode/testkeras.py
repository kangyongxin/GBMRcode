from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

# class MyLayer(layers.Layer):
#     def __init__(self, input_dim=32, unit=32):
#         super(MyLayer, self).__init__()
        
#         w_init = tf.random_normal_initializer()
#         self.weight = tf.Variable(initial_value=w_init(
#             shape=(input_dim, unit), dtype=tf.float32), trainable=True)
        
#         b_init = tf.zeros_initializer()
#         self.bias = tf.Variable(initial_value=b_init(
#             shape=(unit,), dtype=tf.float32), trainable=True)
    
#     def call(self, inputs):
#         return tf.matmul(inputs, self.weight) + self.bias
        
# x = tf.ones((3,5))
# my_layer = MyLayer(5, 4)
# out = my_layer(x)
# print(out)

# 采样网络
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon
# 编码器
class Encoder(layers.Layer):
    def __init__(self, latent_dim=32, 
                intermediate_dim=64, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        z_mean = self.dense_mean(h1)
        z_log_var = self.dense_log_var(h1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
        
# 解码器
class Decoder(layers.Layer):
    def __init__(self, original_dim, 
                 intermediate_dim=64, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)
    
# 变分自编码器
class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim=32, 
                intermediate_dim=64, name='encoder', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
    
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                              intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                              intermediate_dim=intermediate_dim)
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        kl_loss = -0.5*tf.reduce_sum(
            z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)
        return reconstructed

class ImEncoder(tf.keras.Model):
    def __init__(self,original_dim,latent_dim=32,intermediate_dim=64,name='ImEncoder',**kwargs):
        super(ImEncoder,self).__init__(name=name,**kwargs)
        
        self.original_dim =  original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                              intermediate_dim=intermediate_dim)
    
    def call(self,inputs):
        z_mean,z_log_var,z = self.encoder(inputs)
        return z
        
# ————————————————
# 版权声明：本文为CSDN博主「Doit_」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/qq_31456593/java/article/details/88605387

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
#print("xtrain",x_train[0])
plt.imshow(x_train[0]) # 显示黑白图像
plt.savefig("x_train0.png")
plt.imshow(x_train[1])
plt.savefig("x_train1.png")
x_train = x_train.reshape(60000, 784).astype('float32') / 255
vae = VAE(784,32,64)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)
x_test0 = x_train[0:1]
x_test1 = x_train[1]
im2state = ImEncoder(784,32,64)
state1 = im2state(x_test0)
# state2 = im2state(x_test1)
print(state1)