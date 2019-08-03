import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Reshape
from keras import regularizers
from keras.layers.merge import add

# BN层
def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _bn_relu_spc(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

############################################################################

# 带有BN层的三维卷积神经网络,BN层位置不一样，一个在CONV前，一个在CONV后，在CONV后的是要开始输入卷积时用的，
# BN在CONV前，是在残差网络中应用的，不过统一称为CONVBN
# CONV + BN + RELU
def _conv_bn_relu_spc(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    # 128,result,result,97
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))

    def f(input):
        conv = Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer, filters=nb_filter,
                      kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))(input)

        return _bn_relu_spc(conv)

    return f

# BN + RELU + CONV
def _bn_relu_conv_spc(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")  # 基于normal分布，normal的默认 scale=0.05
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu_spc(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3),
                      padding=border_mode)(activation)

    return f

##########################################################################################

# 这一层实现了残差和输入层的融合，处理输入以保持和残差一样的尺寸
def _shortcut_spc(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_dim1 = 1
    stride_dim2 = 1
    stride_dim3 = (input._keras_shape[CONV_DIM3] + 1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    print("input shape:", input._keras_shape)
    # input shape: (None,7,7,97,24)
    print('stride_dim3:', stride_dim3)
    # stride_dim3:result
    print('equal_channels:', equal_channels)
    # equal_channels:True 说明输入和残差的卷积核尺寸是一致的，都是24，input._keras_shape[4]，是可以融合的

    # result X result conv if shape is different. Else identity.
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Convolution3D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
                                 kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                 subsample=(stride_dim1, stride_dim2, stride_dim3),
                                 init="he_normal", border_mode="valid",
                                 W_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

# 光谱特征提取模块，是一个模块，两个特征提取模块，模块里面调方法
def _residual_block_spc(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    这是光谱特征提取的第一个模块，block_function是通过使用加载方法的方法
    """

    def f(input):
        # range(result)也只是迭代了1次
        print('repetitions:', repetitions)
        # repetitions: result

        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (1, 1, 2)
            input = block_function(
                nb_filter=nb_filter,
                init_subsample=init_subsample,
                is_first_block_of_first_layer=(is_first_layer and i == 0)
            )(input)
        return input

    return f

# ########  光谱特征提取有两个模块，一个是层数小于34的，一个是层数大于34的
# 具体的光谱特征提取模型，调用这个模型的时候参数是在模块调用里面赋值的
def basic_block_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 1x1 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """

    def f(input):

        # 如果是输入的第一层，只需要CONV就可以，因为前面执行过CONV + BN + RELU
        # 为了填充特征图大小，padding=same，对于卷积扫描的过程使用0填充，保证了输入和输出特征图的一致。
        # 跨越的那个连线是input，经过两个CONVBN结构的才是残差，输入值经过_bn_relu_conv_spc结构两次，都是same，在输出的时候
        # 融合残差和input就可以
        if is_first_block_of_first_layer:
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample,
                           kernel_regularizer=regularizers.l2(0.0001),
                           filters=nb_filter, kernel_size=(1, 1, 7), padding='same')(input)
        else:
            conv1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7,
                                      subsample=init_subsample)(input)

        residual = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7)(conv1)
        return _shortcut_spc(input, residual)

    return f

# 层数大于34的
def bottleneck_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of nb_filter * 4
    """

    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Convolution3D(nb_filter=nb_filter,
                                     kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                     subsample=init_subsample,
                                     init="he_normal", border_mode="same",
                                     W_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                         subsample=init_subsample)(input)

        conv_3_3 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv_1_1)
        residual = _bn_relu_conv_spc(nb_filter=nb_filter * 4, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1)(conv_3_3)
        return _shortcut_spc(input, residual)

    return f

# 空间特征提取模块的入口外部使用的 CONV + BN + RELU
def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))

    def f(input):
        conv = Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))(input)
        return _bn_relu(conv)

    return f

# 空间特征提取函数的内部残差部分使用的 BN + RELU + CONV
def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(
            activation)

    return f

# 空间特征提取的残差部分和输入部分的融合
def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_dim1 = (input._keras_shape[CONV_DIM1] + 1) // residual._keras_shape[CONV_DIM1]
    stride_dim2 = (input._keras_shape[CONV_DIM2] + 1) // residual._keras_shape[CONV_DIM2]
    stride_dim3 = (input._keras_shape[CONV_DIM3] + 1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    print("input shape:", input._keras_shape)
    # 融合是3*3，一定更改尺寸了所以这里对于输入的input一定也要更改尺寸
    shortcut = Conv3D(kernel_initializer="he_normal", strides=(stride_dim1, stride_dim2, stride_dim3),
                      kernel_regularizer=regularizers.l2(0.0001),
                      filters=residual._keras_shape[CHANNEL_AXIS], kernel_size=(1, 1, 1), padding='valid')(input)

    return add([shortcut, residual])

# 空间特征提取模块的外枚举
def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """

    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2, 1)
            input = block_function(
                nb_filter=nb_filter,
                init_subsample=init_subsample,
                is_first_block_of_first_layer=(is_first_layer and i == 0)
            )(input)
        return input

    return f

# 空间特征提取的具体模型，层数小于34的
def basic_block(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """

    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample,
                           kernel_regularizer=regularizers.l2(0.0001),
                           filters=nb_filter, kernel_size=(3, 3, 1), padding='same')(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1,
                                  subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv1)
        return _shortcut(input, residual)

    return f

# 层数大于34的
# def bottleneck(nb_filter, init_subsample=(result, result, result), is_first_block_of_first_layer=False):
#     """Bottleneck architecture for > 34 layer resnet.
#     Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
#     Returns:
#         A final conv layer of nb_filter * 4
#     """
#
#     def f(input):
#
#         if is_first_block_of_first_layer:
#             # don't repeat bn->relu since we just did bn->relu->maxpool
#             conv_1_1 = Convolution3D(nb_filter=nb_filter,
#                                      kernel_dim1=result, kernel_dim2=result, kernel_dim3=result,
#                                      subsample=init_subsample,
#                                      init="he_normal", border_mode="same",
#                                      W_regularizer=l2(0.0001))(input)
#         else:
#             conv_1_1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=result, kernel_dim2=result, kernel_dim3=result,
#                                      subsample=init_subsample)(input)
#
#         conv_3_3 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=result)(conv_1_1)
#         residual = _bn_relu_conv(nb_filter=nb_filter * 4, kernel_dim1=result, kernel_dim2=result, kernel_dim3=result)(conv_3_3)
#         return _shortcut(input, residual)
#
#     return f

# Keras支持的tensorflow或者theno的维度数上有不同
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

# 可能是在类中加载其他方法的一种方式，传函数名，没传函数本身
def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

# 组合模型
class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn_spc, block_fn, repetitions1, repetitions2):
        # ResnetBuilder.build(input_shape, num_outputs, basic_block_spc, basic_block, [result], [result])
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """

        # 输入是四个维度信息
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)
        # orignal input shape: result,7,7,200

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        print('change input shape:', input_shape)
        # change input shape: 7,7,200,result

        # Load function from str if needed.
        # 通过函数名加载类外的方法
        block_fn_spc = _get_block(block_fn_spc)
        block_fn = _get_block(block_fn)

        # 张量流输入
        input = Input(shape=input_shape)
        print(input)
        # Tensor("input_1:0", shape=(?, 7, 7, 200, result), dtype=float32)
        print('#' * 30)
        print("input shape result:", input._keras_shape[0])
        print("input shape 2:", input._keras_shape[1])
        print("input shape 3:", input._keras_shape[2])
        print("input shape 4:", input._keras_shape[3])
        print("input shape 5:", input._keras_shape[4])
        # input shape result: None
        # input shape 2: 7
        # input shape 3: 7
        # input shape 4: 200
        # input shape 5: result

        # CONV + BN +RELU
        # 提取光谱特征，第一个卷积核数目是24,result*result*7，步长是1*result*2
        conv1_spc = _conv_bn_relu_spc(nb_filter=24, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7, subsample=(1, 1, 2))(
            input)
        print('conv1_spc:', conv1_spc)
        # conv1_spc: Tensor("activation_1/Relu:0", shape=(?, 7, 7, 97, 24), dtype=float32)
        # 卷积核出来是24,空间上是1*result，所以出来还是7*7，光谱上(200-7+result)/2=97,3D的卷积核和这个维度是单独维度，不和
        # 2D一样是直接更改3基色维度的

        block_spc = conv1_spc

        # 结构在这里有两个外层的模块，所以有个循环每个模块内部有两个CONVNB，所以有一个循环
        # 提取光谱特征的残差模块，有两个
        nb_filter = 24
        # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        # repetitions=[result]  i,r = 0,result
        for i, r in enumerate(repetitions1):
            # _residual_block_spc是一个模块里面调用方法，调用的方法就是你传进去的方法block_fn_spc，
            # 而这个方法是在类里面通过调用类外函数传进来的
            # ResnetBuilder.build(input_shape, num_outputs, basic_block_spc, basic_block, [result], [result])
            # build的第三个参数，是具体的光谱特征提取方法，i=0，r=result
            block_spc = _residual_block_spc(block_fn_spc, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(
                block_spc)
            nb_filter *= 2
            # 加倍没有意义，下面两个BN也没用到卷积核

        # 在光谱的输出层使用了2次BN，是符合论文中BN增加正则化从而提高SSRN精度的说法
        # Last activation
        # BN + RELU
        block_spc = _bn_relu_spc(block_spc)

        block_norm_spc = BatchNormalization(axis=CHANNEL_AXIS)(block_spc)
        block_output_spc = Activation("relu")(block_norm_spc)
        # 至此光谱特征提取结束，实际上只有一次残差，并不是两次。

        # (None,7,7,97,24)
        #####################################################################################################
        # 128，进入空间特征提取，CONV + BN + RELU
        conv_spc_results = _conv_bn_relu_spc(nb_filter=128, kernel_dim1=1, kernel_dim2=1,
                                             kernel_dim3=block_output_spc._keras_shape[CONV_DIM3])(block_output_spc)

        print('block_output_spc.kernel_dim3:', block_output_spc._keras_shape[CONV_DIM3])
        # block_output_spc.kernel_dim3: 97 , 这是一个 result,result,97
        print("conv_spc_result shape:", conv_spc_results._keras_shape)
        # conv_spc_result shape: (None, 7, 7, result, 128)

        conv2_spc = Reshape((conv_spc_results._keras_shape[CONV_DIM1], conv_spc_results._keras_shape[CONV_DIM2],
                             conv_spc_results._keras_shape[CHANNEL_AXIS], 1))(conv_spc_results)
        print(conv2_spc)
        # 从(None, 7, 7, result, 128)到(None, 7, 7, 128, result)这种reshape，注意参数，第一个是7，第二个是7，第三个里面指的4的位置
        # 所以是128，最后一个是1.
        # Tensor("reshape_1/Reshape:0", shape=(?, 7, 7, 128, result), dtype=float32)

        # 从reshape这里转到输入网络中的尺寸
        conv1 = _conv_bn_relu(nb_filter=24, kernel_dim1=3, kernel_dim2=3, kernel_dim3=128,
                              subsample=(1, 1, 1))(conv2_spc)
        print("conv1 shape:", conv1._keras_shape)
        # conv1 shape: (None, 5, 5, result, 24)
        # 输入滑窗尺寸是3*3*128，24个卷积核, (7+3)/2=5, 第三维输入是128，尺寸也是128，所以输出是1，核还是24

        block = conv1
        # 进入空间特征提取模块
        nb_filter = 24
        for i, r in enumerate(repetitions2):
            block = _residual_block(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block)
            nb_filter *= 2

        # Last activation
        # 这里应该是 batch_normalization_8
        block = _bn_relu(block)

        block_norm = BatchNormalization(axis=CHANNEL_AXIS)(block)
        block_output = Activation("relu")(block_norm)

        ###################################################################################################
        # 输入分类器
        # Classifier block
        pool2 = AveragePooling3D(pool_size=(block._keras_shape[CONV_DIM1],
                                            block._keras_shape[CONV_DIM2],
                                            block._keras_shape[CONV_DIM3],),
                                 strides=(1, 1, 1))(block_output)
        flatten1 = Flatten()(pool2)
        drop1 = Dropout(0.5)(flatten1)
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(drop1)

        model = Model(inputs=input, outputs=dense)
        model.summary()
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        # (result,7,7,200),16
        return ResnetBuilder.build(input_shape, num_outputs, basic_block_spc, basic_block, [1], [1])  # [2, 2, 2, 2]

    @staticmethod
    def build_resnet_12(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block_spc, basic_block, [2], [2])

def main():
    model = ResnetBuilder.build_resnet_8((1, 7, 7, 200),
                                         16)  # IN DATASET model = ResnetBuilder.build_resnet_18((3, 224, 224), 1000)
    # model = ResnetBuilder.build_resnet_6((result,7,7,176), 13)               # KSC DATASET
    # model = ResnetBuilder.build_resnet_6((result, 7, 7, 103), 9)             # UP DATASET
    # model = ResnetBuilder.build_resnet_34((result, 27, 27, 103), 9)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()

if __name__ == '__main__':
    main()
