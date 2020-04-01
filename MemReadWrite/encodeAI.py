# 构造用来编码的智能体
# 仿照rma.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sonnet as snt
import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


#这里有两种类型的编码器，暂时使用第二种，20200401
# 图像编解码器的构造 1 像素卷积编解码 by sonnet
class ImageEncoderDecoder(snt.Module):
  """Image Encoder/Decoder module."""
  def __init__(
      self,
      image_code_size,
      name='image_encoder_decoder'):
    """Initialize the image encoder/decoder."""
    super(ImageEncoderDecoder, self).__init__(name=name)

    # This is set by a call to `encode`. `decode` will fail before this is set.
    self._convnet_output_shape = None

   
    self._convnet = snt.Conv2D(
        output_channels=(16, 32),
        kernel_shape=(3, 3),
        stride=(1, 1),
        padding=('SAME',))
    self._post_convnet_layer = snt.Linear(image_code_size, name='final_layer')
    print("ImageEncoderDecoder init")
  #@snt.reuse_variables
  def encode(self, image):
    """Encode the image observation."""

    convnet_output = self._convnet(image)

    # Store unflattened convnet output shape for use in decoder.
    self._convnet_output_shape = convnet_output.shape[1:]

    # Flatten convnet outputs and pass through final layer to get image code.
    print("Encode the image observation.")
    return self._post_convnet_layer(snt.Flatten()(convnet_output))

  #@snt.reuse_variables
  def decode(self, code):
    """Decode the image observation from a latent code."""
    if self._convnet_output_shape is None:
      raise ValueError('Must call `encode` before `decode`.')
    transpose_convnet_in_flat = snt.Linear(
        self._convnet_output_shape.num_elements(),
        name='decode_initial_linear')(
            code)
    transpose_convnet_in_flat = tf.nn.relu(transpose_convnet_in_flat)
    transpose_convnet_in = snt.Reshape(
        self._convnet_output_shape.as_list())(transpose_convnet_in_flat)
    print("Decode the image observation from a latent code.")
    return self._convnet.transpose(None)(transpose_convnet_in)

  def _build(self, *args):  # Unused. Use encode/decode instead.
    raise NotImplementedError('Use encode/decode methods instead of __call__.')

# 图像编解码器的构造 2 VAE by keras
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

# 编码并输出
class ImEncoder(tf.keras.Model):
    def __init__(self,original_dim,latent_dim=32,intermediate_dim=64,name='ImEncoder',**kwargs):
        super(ImEncoder,self).__init__(name=name,**kwargs)
        
        self.original_dim =  original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                              intermediate_dim=intermediate_dim)
    
    def call(self,inputs):
        z_mean,z_log_var,z = self.encoder(inputs)
        return z

class Agent():
    def __init__(self,num_actions=None,image_code_size=500,obs_size=75,latent_dim=32,name='encode_agent',**kwargs):
        #super(Agent, self).__init__(name=name,**kwargs)#如果继承别人的才用
        #latent dim 和 image_code_size是相同的含义，用在不同的编码器中
        self.num_actions= num_actions
        self._image_code_size = image_code_size
        self._image_encoder_decoder = ImageEncoderDecoder(image_code_size=image_code_size)
        self._name = name
        self._im2state = ImEncoder(obs_size,latent_dim,64)
        self._vae = VAE(obs_size,latent_dim,64)

    def TakeRandomAction(self):
        action = random.randint(0,self.num_actions-1) 
        return action

    def obs2state(self,observation):
        obs_code = self._image_encoder_decoder.encode(observation)
        return obs_code

    def state2obs(self,state):
        image_recon = self._image_encoder_decoder.decode(state)
        return image_recon
        
    def step(self,observation):
        with tf.name_scope(self._name + '/step'):
            state = self.obs2state(observation)
        
        return state

    def recon_loss(self,observation,image_recon):
        pass
