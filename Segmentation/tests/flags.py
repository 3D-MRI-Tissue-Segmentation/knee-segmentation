from absl import flags

# Dataset/training options
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('batch_size', 32, 'Batch size per TPU Core / GPU')
flags.DEFINE_float('base_learning_rate', 3.2e-04, 'base learning rate at the start of training session')
flags.DEFINE_float('min_learning_rate', 1e-09, 'minimum learning rate')
flags.DEFINE_integer('lr_warmup_epochs', 1, 'No. of epochs for a warmup to the base_learning_rate. 0 for no warmup')
flags.DEFINE_float('lr_drop_ratio', 0.8, 'Amount to decay the learning rate')
flags.DEFINE_bool('custom_decay_lr', False, 'Whether to specify epochs to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', [10, 20, 40, 60], 'Epochs to decay the learning rate by. Only used if custom_decay_lr is True')
flags.DEFINE_string('dataset', 'oai_challenge', 'Dataset: oai_challenge, isic_2018 or oai_full')
flags.DEFINE_bool('use_2d', True, 'True to train on 2D slices, False to train on 3D data')
flags.DEFINE_integer('crop_size', 288, 'Height and width crop size.')
flags.DEFINE_integer('depth_crop_size', 160, 'Depth crop size.')
flags.DEFINE_integer('train_epochs', 50, 'Number of training epochs.')
flags.DEFINE_list('aug_strategy', None, 'Augmentation Strategies: None, random-crop, noise, crop_and_noise')

# Model options
flags.DEFINE_string('model_architecture', 'unet', 'unet, r2unet, segnet, unet++, 100-Layer-Tiramisu, deeplabv3, deeplabv3_plus')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_bool('multi_class', True, 'Whether to train on a multi-class (Default) or binary setting')
flags.DEFINE_integer('kernel_size', 3, 'kernel size to be used')
flags.DEFINE_bool('use_batchnorm', True, 'Whether to use batch normalisation')
flags.DEFINE_bool('use_bias', True, 'Wheter to use bias')
flags.DEFINE_string('channel_order', 'channels_last', 'channels_last (Default) or channels_first')
flags.DEFINE_string('activation', 'relu', 'activation function to be used')
flags.DEFINE_bool('use_dropout', False, 'Whether to use dropout')
flags.DEFINE_bool('use_spatial', False, 'Whether to use spatial Dropout. Only used if use_dropout is True')
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate. Only used if use_dropout is True')
flags.DEFINE_string('optimizer', 'adam', 'Which optimizer to use for model: adam, rmsprop, sgd')

# UNet parameters
flags.DEFINE_list('num_filters', [64, 128, 256, 512, 1024], 'number of filters in the model')
flags.DEFINE_integer('num_conv', 2, 'number of convolution layers in each block')
flags.DEFINE_string('backbone_architecture', 'default', 'default, vgg16, vgg19, resnet50, resnet101, resnet152.')
flags.DEFINE_bool('use_transpose', False, 'Whether to use transposed convolution or upsampling + convolution')
flags.DEFINE_bool('use_attention', False, 'Whether to use attention mechanism')

# 100-layer Tiramisu parameters
flags.DEFINE_list('layers_per_block', [4, 5, 7, 10, 12, 15], 'number of convolutional layers per block')
flags.DEFINE_integer('growth_rate', 16, 'number of feature maps increase after each convolution')
flags.DEFINE_integer('pool_size', 2, 'pooling filter size to be used')
flags.DEFINE_integer('strides', 2, 'strides size to be used')
flags.DEFINE_string('padding', 'same', 'padding mode to be used')
flags.DEFINE_integer('init_num_channels', 48, 'Initial number of filters for the first convolutional layer')

# Deeplab parameters
flags.DEFINE_bool('use_nonlinearity', True, 'Whether to use an activation')
flags.DEFINE_integer('kernel_size_initial_conv', 3, 'kernel size for the first convolution')
flags.DEFINE_integer('num_filters_atrous', 256, 'number of filters for the atrous convolution block')
flags.DEFINE_list('num_filters_DCNN', [256, 512, 1024], 'number of filters for the first three blocks of the DCNN')
flags.DEFINE_integer('num_filters_ASPP', 256, 'number of filters for the ASPP term')
flags.DEFINE_integer('kernel_size_atrous', 3, 'kernel size for the atrous convolutions')
flags.DEFINE_list('kernel_size_DCNN', [1, 3], 'kernel sizes for the blocks of the DCNN')
flags.DEFINE_list('kernel_size_ASPP', [1, 3, 3, 3], 'kernel size for the ASPP term')
flags.DEFINE_list('MultiGrid', [1, 2, 4], 'relative convolution rates for the atrous convolutions')
flags.DEFINE_list('rate_ASPP', [1, 4, 6, 12], 'rates for the ASPP term convolutions')
flags.DEFINE_integer('output_stride', 16, 'final output stride (taking into account max pooling)')

flags.DEFINE_integer('num_filters_final_encoder', 512, 'Number of filters of the last convolution of the encoder')
flags.DEFINE_list('num_filters_from_backbone', [128, 96], 'Number of filters for the 1x1 convolutions to reshape input from the backbone')
flags.DEFINE_list('num_channels_UpConv', [512, 256, 128], 'Number of filters for the upsampling convolutions in the decoder')
flags.DEFINE_integer('kernel_size_UpConv', 3, 'Kernel size for the upsampling convolutions')

# Logging, saving and testing options
flags.DEFINE_string('tfrec_dir', 'gs://oai-ml-dataset/tfrecords/', 'directory for TFRecords folder')
# flags.DEFINE_string('logdir', 'gs://oai-ml-dataset/checkpoints', 'directory for checkpoints')
flags.DEFINE_string('logdir', './', 'directory for checkpoints')
flags.DEFINE_string('weights_dir', 'checkpoints', 'directory for saved model or weights. Only used if train is False')
flags.DEFINE_string('bucket', 'oai-ml-dataset', 'GCloud Bucket for storage of data and weights')
flags.DEFINE_integer('visual_save_freq', 1, 'Save visualisations every x epochs')
flags.DEFINE_integer('roi_npy', 80, 'Save the middle x*x*x voxels')

flags.DEFINE_string('fig_dir', 'figures', 'directory for saved figures')
flags.DEFINE_bool('train', True, 'If True (Default), train the model. Otherwise, test the model')
flags.DEFINE_string('visual_file', None, 'If not None, creates a visual of the model for the time stamp provided.')
flags.DEFINE_string('gif_directory', None, 'Directory of where to put the gif')
flags.DEFINE_integer('gif_epochs', 1000, 'Epochs to include in the creation of the gifs')
flags.DEFINE_string('gif_cmap', 'gray', 'Color map of the gif')
flags.DEFINE_integer('gif_slice', 80, 'Slice that is taken into consideration for the gif')
flags.DEFINE_integer('gif_volume', 1, 'Which volume from the validation dataset to consider')
flags.DEFINE_bool('clean_gif', False, 'False includes text representing epoch number')
flags.DEFINE_string('tpu_dir', None, 'If loading visual file from a tpu other than the tpu you are training with.')
flags.DEFINE_string('which_representation', None, 'Whether to do epoch gif ("epoch") or volume gif ("volume") or "slice"')

# Accelerator flags
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', 'oai-tpu', 'Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS
