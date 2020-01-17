import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from src.utils.training_utils import cross_entropy_loss
from unet import UNet

class Conv1x1Decoder(tf.keras.Model):
    """A stack of 1x1 convolutions that takes two tensors to be concatenated along their channel axes."""
    
    def __init__(self,
                 num_channels, 
                 num_classes,
                 num_1x1_convs,
                 nonlinearity=tf.keras.activations.relu,
                 data_format='channels_last',
                 name='conv_decoder'):

        super(Conv1x1Decoder, self).__init__(name=name)
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_1x1_convs = num_1x1_convs
        self.nonlinearity = nonlinearity
        self.data_format = data_format
        
        if data_format == 'channels_last':
            self.channel_axis = -1
            self.spatial_axes = [1,2]
        elif data_format == 'channels_first':
            self.channel_axis = 1
            self.spatial_axes = [2,3]

    def call(self, features, z):

        shape = tf.keras.backend.shape(features)
        spatial_shape = [shape[axis] for axis in self.spatial_axes]
        multiples = [1] + spatial_shape
        multiples.insert(self.channel_axis, 1)
        
        if len(tf.keras.backend.shape(z)) == 2:
            z = tf.keras.backend.expand_dims(z, axis=2)
            z = tf.keras.backend.expand_dims(z, axis=2)

        # broadcast latent vector to spatial dimensions of the image/feature tensor
        broadcast_z = tf.keras.backend.tile(z, multiples)
        features = tf.keras.backend.concatenate([features, broadcast_z], axis=self.channel_axis)
        for _ in range(self.num_1x1_convs):
            features = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(1,1), strides=1, data_format=self.data_format, activation = self.nonlinearity)
        
        logits = tf.keras.layers.Conv2D(self.num_classes, kernel_size=(1,1), strides=1, data_format=self.data_format, activation = self.nonlinearity)
        
        return logits(features)

class AxisAlignedConvGaussian(tf.keras.Model):
    """A CNN that parametrises a multivariate Gaussian distribution with axis aligned covariance matrix."""

    def __init__(self, 
                 latent_dim,
                 num_channels,
                 nonlinearity=tf.keras.activations.relu,
                 num_convs_per_block=3,
                 data_format='channels_last',
                 name="conv_dist"):
        super(AxisAlignedConvGaussian, self).__init__(name=name)

        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.nonlinearity = nonlinearity
        self.num_convs_per_block = num_convs_per_block
        self.data_format = data_format

        if data_format == 'channels_last':
            self.channel_axis = -1
            self.spatial_axes = [1,2]
        elif data_format == 'channels_first':
            self.channel_axis = 1
            self.spatial_axes = [2,3]

        """quasi-VGGNet model architecture: https://arxiv.org/pdf/1409.1556.pdf."""
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, num_convs_per_block, activation = nonlinearity, padding='same', data_format=data_format),
            tf.keras.layers.Conv2D(64, num_convs_per_block, activation = nonlinearity, padding='same', data_format=data_format),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)), 
            tf.keras.layers.Conv2D(128, num_convs_per_block, activation = nonlinearity, padding='same', data_format=data_format),
            tf.keras.layers.Conv2D(128, num_convs_per_block, activation = nonlinearity, padding='same', data_format=data_format),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)), 
            tf.keras.layers.Conv2D(256, num_convs_per_block, activation = nonlinearity, padding='same', data_format=data_format),
            tf.keras.layers.Conv2D(256, num_convs_per_block, activation = nonlinearity, padding='same', data_format=data_format),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)), 
            tf.keras.layers.Conv2D(512, num_convs_per_block, activation = nonlinearity, padding='same', data_format=data_format),
            tf.keras.layers.Conv2D(512, num_convs_per_block, activation = nonlinearity, padding='same', data_format=data_format),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        ]) 

    def call(self, img, seg=None):

        if seg is not None:
            seg = tf.keras.backend.cast(seg, tf.float32)
            img = tf.keras.backend.concatenate([img, seg], axis=self.channel_axis)
        
        encoding = self.encoder(img)[-1]
        encoding = tf.keras.backend.mean(encoding, axis=self.spatial_axes, keepdims=True)

        mu_log_sigma = tf.keras.layers.Conv2D(2*self.latent_dim, (1,1), stride=1, data_format=self.data_format)(encoding)
        mu_log_sigma = tf.keras.backend.squeeze(mu_log_sigma, axis=self.spatial_axes)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.keras.backend.exp(log_sigma))

class ProbUNet(tf.keras.Model):
    """Probabilistic version of the U-Net model 
    https://arxiv.org/pdf/1806.05034.pdf"""

    def __init__(self,
                 latent_dim,
                 num_channels,
                 num_classes,
                 num_1x1_convs=3,
                 nonlinearity=tf.keras.activations.relu,
                 num_convs_per_block=3,
                 dropout_rate = 0.25,
                 use_spatial_dropout = True,
                 data_format='channels_last',
                 name='prob_unet'):
        super(ProbUNet, self).__init__(name=name)
        
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_1x1_convs = num_1x1_convs
        self.nonlinearity = nonlinearity
        self.num_convs_per_block = num_convs_per_block
        self.dropout_rate = dropout_rate
        self.use_spatial_dropout = use_spatial_dropout
        self.data_format = data_format

        self.unet = UNet(num_channels=num_channels, num_classes=num_classes, nonlinearity=nonlinearity, 
                         num_convs_per_block=num_convs_per_block, dropout_rate=dropout_rate, 
                         use_spatial_dropout=use_spatial_dropout, data_format=data_format)
        self.f_decoder = Conv1x1Decoder(num_channels=num_channels[0], num_classes=num_classes, num_1x1_convs=num_1x1_convs,
                                         nonlinearity=nonlinearity, data_format=data_format)
        self.prior_net = AxisAlignedConvGaussian(latent_dim=latent_dim, num_channels=num_channels, 
                                                 nonlinearity=nonlinearity, num_convs_per_block=num_convs_per_block,
                                                 data_format=data_format, name='prior_net')
        self.posterior_net = AxisAlignedConvGaussian(latent_dim=latent_dim, num_channels=num_channels, 
                                                 nonlinearity=nonlinearity, num_convs_per_block=num_convs_per_block,
                                                 data_format=data_format, name='posterior_net')                                         
        
    def call(self, img, seg=None, is_training=True, one_hot_labels=True):

        if is_training:
            if seg is not None:
                if not one_hot_labels:
                    if self.data_format == 'channels_last':
                        spatial_shape = img.get_shape()[-3:1]
                        one_hot_shape = (-1,) + tuple(spatial_shape) + (self.num_classes,)
                        class_axis = 3
                    elif self.data_format == 'channels_first':
                        spatial_shape = img.get_shape()[-2:]
                        class_axis = 1
                        one_hot_shape = (-1, self.num_classes) + tuple(spatial_shape)

                    seg = tf.reshape(seg, shape=[-1])
                    seg = tf.one_hot(indices=seg, depth=self.num_classes, axis=class_axis)
                    seg = tf.reshape(seg, shape=one_hot_shape)
                seg -= 0.5
            self.q = self.posterior_net(img, seg)
        
        self.p = self.prior_net(img)
        self.unet_features = self.unet(img)

    def reconstruct(self, use_posterior_mean=False, z_q=None):

        if use_posterior_mean:
            z_q = self.q.loc
        else:
            if z_q is None:
                z_q = self.q.sample()
        
        return self.f_decoder(self.unet_features, z_q)

    def sample(self):

        z_p = self.p.sample()

        return self.f_decoder(self.unet_features, z_p)

    def kl_divergence(self, automatic=True, z_q=None):

        if automatic:
            kl = tfd.kl_divergence(self.q, self.p)
        else:
            if z_q is None:
                z_q = self.q.sample()
            log_q = self.q.log_prob(z_q)
            log_p = self.p.log_prob(z_q)

            kl = log_q - log_p
        
        return kl

    def elbo(self, seg, beta=1.0, automatic_kl=True, reconstruct_posterior_mean=False, z_q=None, one_hot_labels=True, 
             loss_mask=None):

        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        
        if z_q is None:
            z_q = self.q.sample()

        self.kl_loss = tf.reduce_mean(self.kl_divergence(automatic_kl, z_q))
        
        self.rec_logits = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, z_q=z_q)
        
        reconstruction_loss = cross_entropy_loss(labels=seg, logits=self.rec_logits, n_classes=self.num_classes, 
                                                 loss_mask=loss_mask, one_hot_labels=one_hot_labels)
        
        self.reconstruction_loss = reconstruction_loss['sum']
        self.reconstruction_loss_mean = reconstruction_loss['mean']

        return -(self.reconstruction_loss + beta * self.kl_loss)

