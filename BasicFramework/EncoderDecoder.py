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