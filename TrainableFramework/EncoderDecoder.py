# 构造基本的编解码器
# 图像编解码器的构造 2 VAE by keras

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon
# 2.1 编码器
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
        
# 2.2 解码器
class Decoder(layers.Layer):
    def __init__(self, original_dim, 
                 intermediate_dim=64, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)
    
# 2.3 变分自编码器
class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim=32, 
                intermediate_dim=64, name='vae', **kwargs):
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

# 2.4 编码并输出
class ImEncoder(tf.keras.Model):
    def __init__(self,original_dim,latent_dim=32,intermediate_dim=64,name='ImEncoder',**kwargs):
        super(ImEncoder,self).__init__(name=name,**kwargs)
        
        self.original_dim =  original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                              intermediate_dim=intermediate_dim)
    
    def call(self,inputs):
        z_mean,z_log_var,z = self.encoder(inputs)
        return z

# 2.5 解码并输出
class ImDecoder(tf.keras.Model):
    def __init__(self,original_dim,latent_dim=32,intermediate_dim=64,name='ImDecoder',**kwargs):
        super(ImDecoder,self).__init__(name=name,**kwargs)
        
        self.original_dim =  original_dim
        self.decoder = Decoder(original_dim=original_dim,
                              intermediate_dim=intermediate_dim)
    
    def call(self,inputs):
        ReconstructedIm= self.encoder(inputs)
        return ReconstructedIm


# 向值函数解码
class VDecoder(tf.keras.Model):
    '''
    从状态z解码到v值，用了隐含层200个节点的MLP,仿照tvt rma.py 中policy 构造baseline的做法
    需要进一步思考的是如果我们把所有的v值都融合到节点属性了，那么这个值函数的预测是否还有意义，
    或者说我们在图结构中写入的值函数是否有必要是这个值函数。
    '''
    def __init__(self, v_dim, 
                 intermediate_dim=200, name='vdecoder', **kwargs):
        super(VDecoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='tanh')
        self.dense_output = layers.Dense(v_dim, activation='tanh')
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)


# 解码出两个输出，一个是值函数，一个是图像的排列
class VAEV(tf.keras.Model):
    def __init__(self, original_dim, latent_dim=32, 
                intermediate_dim=64, name='vaev', **kwargs):
        super(VAEV, self).__init__(name=name, **kwargs)
    
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                              intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                              intermediate_dim=intermediate_dim)
        self.vdecoder = VDecoder(1,200)
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        v_hat = self.vdecoder(z)
        
        kl_loss = -0.5*tf.reduce_sum(
            z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)
        # self.vaev = keras.Model(
        #     inputs  = inputs,
        #     outputs = [reconstructed,v_hat],
        # )
        return reconstructed,v_hat


#尝试另外一种编解码的结构，就是直接对图像进行编解码 tvt中用的就是这种，把它用keras 实现了
