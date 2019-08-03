from keras.models import Model
from keras.layers import Reshape, Dense, multiply, Permute
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    Concatenate,
    GlobalAveragePooling3D)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D
)
from keras import backend as K


def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4


def _convbnrelu(x, nb_filters, stride, kernel_size, name):
    """
    Convolution block of the first layer

    :param x: input tensor
    :param nb_filters: integer or tuple, number of filters
    :param stride: integer or tuple, stride of convolution
    :param kernel_size: integer or tuple, filter's kernel size
    :param name: string, block label

    :return: output tensor of a block
    """
    x = Conv3D(filters=nb_filters, strides=stride, kernel_size=kernel_size, padding='same',
               kernel_initializer='he_normal', use_bias=False, name=name + "_conv3d", )(x)
    x = BatchNormalization(name=name + "_batch_norm")(x)
    x = Activation(activation='relu', name=name + '_relu')(x)
    return x


def _bottleneck(x, growth_rate, stride, name):
    """
    DenseNet-like block for subsequent layers
    :param x: input tensor
    :param growth_rate: integer, number of output channels
    :param stride: integer, stride of 3x3 convolution
    :param name: string, block label

    :return: output tensor of a block
    """
    x = Conv3D(filters=4 * growth_rate, strides=1, kernel_size=1, padding='same',
               kernel_initializer='he_normal', use_bias=False, name=name + "_conv3d_1x1x1")(x)
    x = BatchNormalization(name=name + "_batch_norm_1")(x)
    x = Activation(activation='relu', name=name + "_relu_1")(x)
    x = Conv3D(filters=growth_rate, strides=stride, kernel_size=3,
               padding='same', kernel_initializer='he_normal', use_bias=False,
               name=name + "_conv3d_3x3x3")(x)
    x = BatchNormalization(name=name + "_batch_norm_2")(x)
    x = Activation(activation='relu', name=name + "_relu_2")(x)
    return x


def basic_block(x, l_growth_rate=None, scale=3, name="basic_block"):
    """
    Basic building block of MSDNet

    :param x: Input tensor or list of tensors
    :param l_growth_rate: list, numbers of output channels for each scale
    :param scale: Number of different scales features
    :param name:
    :return: list of different scales features listed from fine-grained to coarse
    """
    output_features = []

    try:
        is_tensor = K.is_keras_tensor(x)
        # check if not a tensor
        # if keras/tf class raise error instead of assign False
        if not is_tensor:
            raise TypeError("Tensor or list [] expected")

    except ValueError:
        # if not keras/tf class set False
        is_tensor = False

    if is_tensor:

        for i in range(scale):
            mult = 2 ** i
            x = _convbnrelu(x, nb_filters=32 * mult, stride=min(2, mult), kernel_size=3, name=name + "_" + str(i))
            output_features.append(x)

    else:

        assert len(l_growth_rate) == scale, "Must be equal: len(l_growth_rate)={0} scale={1}".format(len(l_growth_rate),
                                                                                                     scale)

        for i in range(scale):
            if i == 0:
                conv = _bottleneck(x[i], growth_rate=l_growth_rate[i], stride=1,
                                   name=name + "_conv3d_" + str(i))
                bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
                conc = Concatenate(axis=bn_axis, name=name + "_concat_post_" + str(i))([conv, x[i]])
            else:
                strided_conv = _bottleneck(x[i - 1], growth_rate=l_growth_rate[i], stride=2,
                                           name=name + "_strided_conv3d_" + str(i))

                conv = _bottleneck(x[i], growth_rate=l_growth_rate[i], stride=1,
                                   name=name + "_conv3d_" + str(i))
                bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
                conc = Concatenate(axis=bn_axis, name=name + "_concat_pre_" + str(i))([strided_conv, conv, x[i]])

            output_features.append(conc)

    return output_features


def transition_block(x, reduction, name):
    """
    Transition block for network reduction
    :param x: list, set of tensors
    :param reduction: float, fraction of output channels with respect to number of input channels
    :param name: string, block label

    :return: list of tensors
    """
    output_features = []
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    for i, item in enumerate(x):
        conv = _convbnrelu(item, nb_filters=int(reduction * K.int_shape(item)[bn_axis]), stride=1, kernel_size=1,
                           name=name + "_transition_block_" + str(i))
        output_features.append(conv)

    return output_features


def classifier_block(x, nb_filters, nb_classes, activation, name):
    """
    Classifier block
    :param x: input tensor
    :param nb_filters: integer, number of filters
    :param nb_classes: integer, number of classes
    :param activation: string, activation function
    :param name: string, block label

    :return: block tensor
    """
    x = _convbnrelu(x, nb_filters=nb_filters, stride=2, kernel_size=3, name=name + "_1")
    x = _convbnrelu(x, nb_filters=nb_filters, stride=2, kernel_size=3, name=name + "_2")
    x = AveragePooling3D(pool_size=2, strides=2, padding='same', name=name + '_avg_pool3d')(x)
    x = Flatten(name=name + "_flatten")(x)
    out = Dense(units=nb_classes, activation=activation, name=name + "_dense")(x)
    return out


# 组合模型
class ResnetBuilder(object):
    @staticmethod
    def build(input_size, nb_classes=9, scale=3, depth=7, l_growth_rate=(6, 12, 24),
              transition_block_location=(12, 20), classifier_ch_nb=128, classifier_location=(7,)):
        """
        Function that builds MSDNet

        :param input_size: tuple of integers, 3x1, size of input image
        :param nb_classes: integer, number of classes
        :param scale: integer, number of network's scales
        :param depth: integer, network depth
        :param l_growth_rate: tuple of integers, scale x result, growth rate of each scale
        :param transition_block_location: tuple of integer, array of block's numbers to place transition block after
        :param classifier_ch_nb: integer, output channel of conv blocks in classifier, if None than the same number as in
                                          an input tensor
        :param classifier_location: tuple of integers, array of block's numbers to place classifier after

        :return: MSDNet
        """
        print('original input shape:', input_size)
        _handle_dim_ordering()
        if len(input_size) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_size)

        if K.image_dim_ordering() == 'tf':
            input_size = (input_size[1], input_size[2], input_size[3], input_size[0])
        print('change input shape:', input_size)

        inp = Input(shape=input_size)
        out = []

        for i in range(depth):

            if i == 0:
                x = basic_block(inp, l_growth_rate=[],
                                scale=scale, name="basic_block_" + str(i + 1))
            elif i in transition_block_location:
                x = transition_block(x, reduction=0.5, name="transition_block_" + str(i + 1))

                x = basic_block(x, l_growth_rate=l_growth_rate,
                                scale=scale, name="basic_block_" + str(i + 1))
                scale -= 1
                l_growth_rate = l_growth_rate[1:]
                x = x[1:]
            else:
                if i in (1, 2, 3, 4, 5):
                    x = basic_block(x, l_growth_rate=l_growth_rate,
                                    scale=scale, name="basic_block_" + str(i + 1))
                    x = transition_block(x, reduction=0.5, name="transition_block_" + str(i + 1))
                elif i == 6:
                    x = basic_block(x, l_growth_rate=l_growth_rate,
                                    scale=scale, name="basic_block_" + str(i + 1))

            if i + 1 in classifier_location:
                bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
                cls_ch = K.int_shape(x[-1])[bn_axis] if classifier_ch_nb is None else classifier_ch_nb
                out.append(classifier_block(x[-1], nb_filters=cls_ch, nb_classes=nb_classes, activation='sigmoid',
                                            name='classifier_' + str(i + 1)))

        return Model(inputs=inp, outputs=out)

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs)


def main():
    model = ResnetBuilder.build_resnet_8((1, 15, 15, 103), 9)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary(positions=[.33, .61, .71, 1.])


if __name__ == '__main__':
    main()
