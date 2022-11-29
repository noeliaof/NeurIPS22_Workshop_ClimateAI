# TensorFlow ≥2.0 is required
import tensorflow_addons as tfa
import tensorflow as tf
assert tf.__version__ >= '2.0'

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from utils.utils_unet import *
## Adding a second class

class Unet_Inputs(tf.keras.Model):
    """
    Adapted from model factory.
    """

    def __init__(self, arch, input_size, output_size, for_extremes = False,
                 dropout_rate=0.2, use_batch_norm=True, inner_activation='relu', unet_filters_nb=64, 
                 unet_depth=4, unet_use_upsample=True, bottleneck=False, output_scaling=1, input_index = None, output_crop=None):
        
        super(Unet_Inputs, self).__init__()
        self.arch = arch
        self.input_size = list(input_size)
        self.output_size = list(output_size)
        self.input_index = input_index
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.unet_use_upsample = unet_use_upsample
        self.inner_activation = inner_activation
        self.unet_depth = unet_depth
        self.unet_filters_nb = unet_filters_nb
        self.output_scaling = output_scaling
        self.output_crop = output_crop
        self.last_activation = 'relu'
        self.bottleneck = bottleneck
        self.for_extremes = for_extremes
        if (len(output_size) > 2):
            self.out_channels = self.output_size[2]
        else:
            self.out_channels = 1
        print(self.out_channels)
        
        if for_extremes:
            self.last_activation = 'sigmoid'
            
        if arch == 'Unet':
            self.build_Unet()
        elif arch == 'UnetI':
            self.build_UnetI()
        elif arch == 'UnetIndex':
            self.build_UnetIndex()
        else:
            raise ValueError('The architecture was not correctly defined')
      
            
    def build_Unet(self):
        """
        Based on: U-Net: https://github.com/nikhilroxtomar/Unet-for-Person-Segmentation/blob/main/model.py
        """
        
        # Downsampling
        inputs = layers.Input(shape=self.input_size)
        
        # Pad if necessary
        x = self.padding_block(inputs, factor=2**self.unet_depth)
        
        skips = []
        for i in range(self.unet_depth):
            s, x = self.unet_encoder_block(x, self.unet_filters_nb * 2**i)
            skips.append(s)
        
        x = self.conv_block(x, self.unet_filters_nb * 2**self.unet_depth, 3, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, self.unet_filters_nb * 2**self.unet_depth, 3, initializer='he_normal', with_batchnorm=True)
        
        
        # testting adding bottleneck
        if self.bottleneck:
            x1 = Flatten()(x)
            n_shape= x.shape[1:]
            xx = Dense(x1.shape[1], activation="relu")(x1)
            print('adding dense layer')
            x3 = Reshape(n_shape)(xx)
            for i in reversed(range(self.unet_depth)):
                if i == 3:
                    x = self.unet_decoder_block(x3, skips[i], self.unet_filters_nb * 2**i, is_last=(i==0))
                else:
                    x = self.unet_decoder_block(x, skips[i], self.unet_filters_nb * 2**i, is_last=(i==0))
        else:
            print('original design')
            # Upsampling
            for i in reversed(range(self.unet_depth)):
                x = self.unet_decoder_block(x, skips[i], self.unet_filters_nb * 2**i, is_last=(i==0))

        # Additional upsampling for downscaling
        x = self.handle_output_scaling(x)

        x = self.conv_block(x, self.out_channels, 1, activation=self.last_activation)
        outputs = self.final_cropping_block(x)

        self.model = keras.Model(inputs, outputs, name="U-Net")
        
    
    def build_UnetI(self):
        """
        Based on: U-Net:but it includes the temporal index in the bottleneck part
        """
        
        # Downsampling
        inputs = layers.Input(shape=self.input_size)
        
        # Pad if necessary
        x = self.padding_block(inputs, factor=2**self.unet_depth)
        
        skips = []
        for i in range(self.unet_depth):
            s, x = self.unet_encoder_block(x, self.unet_filters_nb * 2**i)
            skips.append(s)
        
        x = self.conv_block(x, self.unet_filters_nb * 2**self.unet_depth, 3, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, self.unet_filters_nb * 2**self.unet_depth, 3, initializer='he_normal', with_batchnorm=True)
        # keep the shape
        n_shape = x.shape[1:]
        
        inputB = layers.Input(shape=self.input_index)
        x1 = Flatten()(x)
        x2 = Dense(self.unet_filters_nb, activation="relu")(inputB)
        xx= layers.Concatenate()([x1, x2])
        xx = Dense(x1.shape[1], activation="relu")(xx)
        xx = Dropout(self.dropout_rate)(xx)
        x3 = Reshape(n_shape)(xx)
        
        for i in reversed(range(self.unet_depth)):
            if i == 3:
                x = self.unet_decoder_block(x3, skips[i], self.unet_filters_nb * 2**i, is_last=(i==0))
            else:
                x = self.unet_decoder_block(x, skips[i], self.unet_filters_nb * 2**i, is_last=(i==0))
 
        # Additional upsampling for downscaling
        x = self.handle_output_scaling(x)

        x = self.conv_block(x, self.out_channels, 1, activation=self.last_activation)
        outputs = self.final_cropping_block(x)
        
        # add the other input
        self.model = keras.Model([inputs, inputB], outputs, name="U-Net-2I")
        
        
       
    def build_UnetIndex(self):
        """
        Based on: U-Net:but it includes the temporal index within the encodes
        """
        
        # Spatial inputs
        inputs = layers.Input(shape=self.input_size)
        # temporal inputs
        inputB = layers.Input(shape=self.input_index)
        # Pad if necessary
        x = self.padding_block(inputs, factor=2**self.unet_depth)
        n_shape = x.shape[1:]
        # Now concatenate them: 
        # flatten and reshape
        xf = Flatten()(x)
        xi = layers.concatenate([xf,inputB])
        # I need to add 2 dense layer
        xx = Dense(self.unet_filters_nb, activation="relu")(xi)
        xx = Dense(xf.shape[1], activation="relu")(xx)
        #xx = Dense(xf.shape[1], activation="relu")(xi) # too much memory
        x = Reshape(n_shape)(xx)
        # now apply conv block
        skips = []
        for i in range(self.unet_depth):
            s, x = self.unet_encoder_block(x, self.unet_filters_nb * 2**i)
            skips.append(s)
        
        x = self.conv_block(x, self.unet_filters_nb * 2**self.unet_depth, 3, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, self.unet_filters_nb * 2**self.unet_depth, 3, initializer='he_normal', with_batchnorm=True)
       
        # now deconv block
        
        # Upsampling
        for i in reversed(range(self.unet_depth)):
            x = self.unet_decoder_block(x, skips[i], self.unet_filters_nb * 2**i, is_last=(i==0))
        
        # Additional upsampling for downscaling
        x = self.handle_output_scaling(x)

        x = self.conv_block(x, self.out_channels, 1, activation=self.last_activation)
        outputs = self.final_cropping_block(x)

        # add the other input
        self.model = keras.Model([inputs, inputB], outputs, name="U-Net-2Index")
        
        
                
        
    def unet_encoder_block(self, input, filters, kernel_size=3):
        x = self.conv_block(input, filters, kernel_size, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, filters, kernel_size, initializer='he_normal', with_batchnorm=True)
        p = layers.MaxPooling2D((2, 2))(x)
        
        return x, p

    
    def unet_decoder_block(self, input, skip_features, filters, conv_kernel_size=3, deconv_kernel_size=2, is_last=False):
        x = self.deconv_block(input, filters, deconv_kernel_size, stride=2)
        x = layers.Concatenate()([x, skip_features])
        x = self.conv_block(x, filters, conv_kernel_size, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, filters, conv_kernel_size, initializer='he_normal', with_batchnorm=(not is_last))

        return x
        
        
    def unet_decoder_convlstm_block(self, input, skip_features, filters, conv_kernel_size=3, deconv_kernel_size=2, is_last=False):
        x = self.deconv_block(input, filters, deconv_kernel_size, stride=2)
        # reshape 
        xx1 = layers.Reshape(target_shape=(1, x.shape[1], x.shape[2], filters))(skip_features)
        xx2 = layers.Reshape(target_shape=(1, x.shape[1], x.shape[2], filters))(x)
        x = layers.Concatenate()([xx1,xx2])
        # now apply CONVLSTM
        x = layers.ConvLSTM2D(filters, conv_kernel_size, padding='same', return_sequences = False, 
                       go_backwards = True,kernel_initializer = 'he_normal' )(x)
        
        x = self.conv_block(x, filters, conv_kernel_size, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, filters, conv_kernel_size, initializer='he_normal', with_batchnorm=(not is_last))

        return x
        
    def conv_block(self, input, filters, kernel_size=3, stride=1, padding='same', initializer='default', activation='default', 
                   with_batchnorm=False, with_pooling=False, with_dropout=False, with_late_activation=False):
        if activation == 'default':
            activation = self.inner_activation
            
        conv_activation = activation
        if with_late_activation:
            conv_activation = None
            
        if initializer == 'default':
            x = layers.Conv2D(filters, kernel_size, strides=(stride, stride), padding=padding, activation=conv_activation)(input)
        else:
            x = layers.Conv2D(filters, kernel_size, strides=(stride, stride), padding=padding, activation=conv_activation, kernel_initializer=initializer)(input)
            
        if with_batchnorm:
            x = layers.BatchNormalization()(x)
        if with_late_activation:
            x = layers.Activation(activation)(x)
        if with_pooling:
            x = layers.MaxPooling2D(pool_size=2)(x)
        if with_dropout:
            x = layers.SpatialDropout2D(self.dropout_rate)(x)
        
        return x
    
    
    def deconv_block(self, input, filters, kernel_size=3, stride=1, padding='same', initializer='default', activation='default', 
                     with_batchnorm=False, with_dropout=False):
        if activation == 'default':
            activation = self.inner_activation
        
        if self.unet_use_upsample:
            x = layers.UpSampling2D((2, 2))(input)
        else:
            if initializer == 'default':
                x = layers.Conv2DTranspose(filters, kernel_size, strides=stride, padding=padding, activation=activation)(input)
            else:
                x = layers.Conv2DTranspose(filters, kernel_size, strides=stride, padding=padding, activation=activation, kernel_initializer=initializer)(input)
            
        if with_batchnorm:
            x = layers.BatchNormalization()(x)
        if with_dropout:
            x = layers.SpatialDropout2D(self.dropout_rate)(x)
        
        return x
    
    
    def dense_block(self, input, units, activation='default', with_dropout=False):
        if activation == 'default':
            activation=self.inner_activation
            
        x = layers.Dense(units, activation=activation)(input)
        if with_dropout:
            x = layers.Dropout(self.dropout_rate)(x)
            
        return x

    
    def handle_output_scaling(self, input, with_batchnorm=False):
        if self.output_scaling > 1:
            if self.output_scaling == 2:
                x = self.deconv_block(input, 64, 3, stride=2, with_batchnorm=with_batchnorm)
            elif self.output_scaling == 3:
                x = self.deconv_block(input, 64, 3, stride=3, with_batchnorm=with_batchnorm)
            elif self.output_scaling == 4:
                x = self.deconv_block(input, 64, 3, stride=2, with_batchnorm=with_batchnorm)
                x = self.deconv_block(x, 64, 3, stride=2, with_batchnorm=with_batchnorm)
            elif self.output_scaling == 5:
                x = self.deconv_block(input, 64, 3, stride=3, with_batchnorm=with_batchnorm)
                x = self.deconv_block(x, 64, 3, stride=2, with_batchnorm=with_batchnorm)
            else:
                raise NotImplementedError('Level of downscaling not implemented')
        else:
            x = input
        
        if self.output_crop:
            raise NotImplementedError('Manual cropping not yet implemented')
            
        return x
            
        
    def padding_block(self, x, factor):
        h, w = x.get_shape().as_list()[1:3]
        dh = 0
        dw = 0
        if h % factor > 0:
            dh = factor - h % factor
        if w % factor > 0:
            dw = factor - w % factor
        if dh > 0 or dw > 0:
            top_pad = dh//2
            bottom_pad = dh//2 + dh%2
            left_pad = dw//2
            right_pad = dw//2 + dw%2
            x = layers.ZeroPadding2D(padding=((top_pad, bottom_pad), (left_pad, right_pad)))(x)
        
        return x
        
        
    def final_cropping_block(self, x):
        # Compute difference between reconstructed width and hight and the desired output size.
        h, w = x.get_shape().as_list()[1:3]
        h_tgt, w_tgt = self.output_size[:2]
        dh = h - h_tgt
        dw = w - w_tgt

        if dh < 0 or dw < 0:
            raise ValueError(f'Negative values in output cropping dh={dh} and dw={dw}')

        # Add to decoder cropping layer and final reshaping
        x = layers.Cropping2D(cropping=((dh//2, dh-dh//2), (dw//2, dw-dw//2)))(x)
        #x = layers.Reshape(target_shape=self.output_size,)(x)
        
        return x
        

    def get_shape_for(self, stride_factor):
        next_shape = self.output_size.copy()
        next_shape[0] = int(np.ceil(next_shape[0]/stride_factor))
        next_shape[1] = int(np.ceil(next_shape[1]/stride_factor))

        return next_shape

        
    def call(self, x):
        return self.model(x)
    
    
    
    
def res_block(x, nb_filters, strides):
    
    res_path = layers.BatchNormalization()(x)
    res_path = layers.Activation(activation='relu')(res_path)
    res_path = layers.Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = layers.BatchNormalization()(res_path)
    res_path = layers.Activation(activation='relu')(res_path)
    res_path = layers.Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = layers.Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = layers.BatchNormalization()(shortcut)

    res_path = layers.add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = layers.BatchNormalization()(main_path)
    main_path = layers.Activation(activation='relu')(main_path)

    main_path = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = layers.BatchNormalization()(shortcut)

    main_path = layers.add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    
   # main_path = UpSampling2D(size=(2, 2))(x)
    main_path =layers.Conv2DTranspose(32, kernel_size=2, strides=2, padding='same', activation='relu')(x)
    # need to crop out
    main_path = crop_output(from_encoder[2],main_path)
    main_path = layers.concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])
   # main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path =layers.Conv2DTranspose(32, kernel_size=2, strides=2, padding='same', activation='relu')(main_path)
    
    main_path = crop_output(from_encoder[1],main_path)
    main_path = layers.concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

   # main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = layers.Conv2DTranspose(32, kernel_size=2, strides=2, padding='same', activation='relu')(main_path)
    main_path = crop_output(from_encoder[0],main_path)
    main_path = layers.concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path


def build_res_unet(input_shape, out_channels, for_extremes = True):
    
    inputs = layers.Input(shape=input_shape)

    to_decoder = encoder(inputs)

    #path = res_block(to_decoder[2], [512, 512], [(1, 1), (1, 1)])
    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder)

    if for_extremes:
        last_activation = 'sigmoid'
    else:
        last_activation = 'relu'
        
    path = layers.Conv2D(filters = out_channels, kernel_size=(1, 1), activation=last_activation)(path) #check activation, it should be here relu/linear

    return Model(inputs, path)


def initiate_optimizer(dg, BATCH_SIZE, lr_method, lr=.000001, init_lr=0.0001, max_lr=0.0001):
    if lr_method == 'Cyclical':
        # Cyclical learning rate
        steps_per_epoch = dg.n_samples // BATCH_SIZE
        clr = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=init_lr,
            maximal_learning_rate=max_lr,
            scale_fn=lambda x: 1/(2.**(x-1)),
            step_size=2 * steps_per_epoch)
        optimizer = tf.keras.optimizers.Adam(clr)
    elif lr_method == 'CosineDecay':
        decay_steps = EPOCHS * (dg.n_samples / BATCH_SIZE)
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            init_lr, decay_steps)
        optimizer = tf.keras.optimizers.Adam(lr_decayed_fn)
    elif lr_method == 'Constant':
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    else:
        raise ValueError('learning rate schedule not well defined.')
        
    return optimizer