# CNNのベースを書き込んでいるプログラム

from keras import Model, regularizers
from keras.models import Sequential
from keras.layers import Input, Flatten, Dropout, Dense, Activation, add, Add, Conv2D, Convolution2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras import regularizers

def _shortcut(inputs, residual):

    # _keras_shape[3] チャンネル数
    n_filters = residual._keras_shape[3]

    # inputs と residual とでチャネル数が違うかもしれない。
    # そのままだと足せないので、1x1 conv を使って residual 側のフィルタ数に合わせている
    shortcut = Convolution2D(n_filters, (1,1), strides=(1,1), padding='valid')(inputs)

    # 2つを足す
    return add([shortcut, residual])


def _resblock(n_filters, strides=(1, 1)):
    def f(input):
        x = Convolution2D(n_filters, (3, 3), strides=strides, kernel_initializer='he_normal', padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(n_filters, (3, 3), strides=strides, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)

        return _shortcut(input, x)

    return f


def _pre_act_resblock(n_filters, strides=(1, 1), decay=0.0001):
    def f(input):
        x = BatchNormalization()(input)
        x = Activation('relu')(x)
        x = Convolution2D(n_filters, (3, 3), strides=strides, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(decay), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(n_filters, (3, 3), strides=strides, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(decay), padding='same')(x)

        return _shortcut(input, x)

    return f


def _bottleneck(n_filters1, n_filters2, strides=(1, 1)):
    def f(input):
        x = Convolution2D(n_filters1, (1, 1), strides=strides, kernel_initializer='he_normal', padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(n_filters1, (3, 3), strides=strides, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(n_filters2, (1, 1), strides=strides, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)

        return _shortcut(input, x)

    return f


def cba(inputs, filters, kernel_size, strides, decay):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(decay), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def plain_net(height, width, channel, n_categories):
    inputs = Input(shape=(height, width, channel))

    x = cba(inputs, filters=16, kernel_size=(7, 7), strides=(2, 2), decay=0.0001)

    x = MaxPooling2D(strides=(2, 2), padding='same')(x)

    x = cba(x, filters=32, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)
    x = cba(x, filters=32, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)

    x = MaxPooling2D(strides=(2, 2), padding='same')(x)

    x = cba(x, filters=64, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)
    x = cba(x, filters=64, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)

    x = MaxPooling2D(strides=(2, 2), padding='same')(x)

    # x = Flatten()(x)
    # x = Dense(1024, kernel_initializer='he_normal', activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1024, kernel_initializer='he_normal', activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_categories, kernel_initializer='he_normal', activation='softmax')(x)

    model = Model(inputs, x)

    return model


def fine_tuned_VGG16(height, width, channel, n_categories, layers):
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(height, width, channel)))

    out_model = Sequential()
    out_model.add(Flatten())
    out_model.add(Dense(4096))
    out_model.add(Dropout(0.5))
    out_model.add(Dense(4096))
    out_model.add(Dropout(0.5))
    out_model.add(Dense(n_categories, activation="softmax"))

    model = Model(inputs=base_model.input, outputs=out_model(base_model.output))
    # model.summary()

    # VGG16の重みを凍結
    for layer in model.layers[:layers]:
        layer.trainable = False

    return model


def VGG16_original(height, width, channel, n_categories):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(height, width, channel)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories, activation="softmax"))

    return model


def VGG16_custom(height, width, channel, n_categories):
    inputs = Input(shape=(height, width, channel))

    x = cba(inputs, filters=16, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)
    x = cba(x, filters=16, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)

    x = MaxPooling2D(strides=(2, 2), padding='same')(x)

    x = cba(x, filters=16, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)
    x = cba(x, filters=16, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)

    x = MaxPooling2D(strides=(2, 2), padding='same')(x)

    x = cba(x, filters=32, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)
    x = cba(x, filters=32, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)
    x = cba(x, filters=32, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)

    x = MaxPooling2D(strides=(2, 2), padding='same')(x)

    x = cba(x, filters=32, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)
    x = cba(x, filters=32, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)
    x = cba(x, filters=32, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)

    x = MaxPooling2D(strides=(2, 2), padding='same')(x)

    x = cba(x, filters=64, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)
    x = cba(x, filters=64, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)
    x = cba(x, filters=64, kernel_size=(3, 3), strides=(1, 1), decay=0.0001)

    # x = MaxPooling2D(strides=(2, 2), padding='same')(x)

    # x = Flatten()(x)
    # x = Dense(1024, kernel_initializer='he_normal', activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1024, kernel_initializer='he_normal', activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D(name='GAP')(x)
    x = Dense(n_categories, kernel_initializer='he_normal', activation='softmax')(x)

    model = Model(inputs, x)

    return model


# 3枚の画像を入力し, GAP層手前で連結.
def VGG16_multi(height, width, channel, n_categories):
    input1 = Input(shape=(height, width, channel))
    input2 = Input(shape=(height, width, channel))
    input3 = Input(shape=(height, width, channel))

    o_kernel = (3, 3)
    h_kernel = (3, 3)
    p_kernel = (3, 3)

    # merge_input = add([input1, input2])
    ''' 1枚目(O) '''
    x1 = cba(input1, filters=16, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = cba(x1, filters=16, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = MaxPooling2D(strides=(2, 2), padding='same')(x1)

    x1 = cba(x1, filters=16, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = cba(x1, filters=16, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = MaxPooling2D(strides=(2, 2), padding='same')(x1)

    x1 = cba(x1, filters=32, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = cba(x1, filters=32, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = cba(x1, filters=32, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = MaxPooling2D(strides=(2, 2), padding='same')(x1)

    x1 = cba(x1, filters=32, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = cba(x1, filters=32, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = cba(x1, filters=32, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = MaxPooling2D(strides=(2, 2), padding='same')(x1)

    x1 = cba(x1, filters=64, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = cba(x1, filters=64, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)
    x1 = cba(x1, filters=64, kernel_size=o_kernel, strides=(1, 1), decay=0.0001)

    ''' 2枚目(H) '''
    x2 = cba(input2, filters=16, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = cba(x2, filters=16, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = MaxPooling2D(strides=(2, 2), padding='same')(x2)

    x2 = cba(x2, filters=16, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = cba(x2, filters=16, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = MaxPooling2D(strides=(2, 2), padding='same')(x2)

    x2 = cba(x2, filters=32, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = cba(x2, filters=32, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = cba(x2, filters=32, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = MaxPooling2D(strides=(2, 2), padding='same')(x2)

    x2 = cba(x2, filters=32, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = cba(x2, filters=32, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = cba(x2, filters=32, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = MaxPooling2D(strides=(2, 2), padding='same')(x2)

    x2 = cba(x2, filters=64, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = cba(x2, filters=64, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)
    x2 = cba(x2, filters=64, kernel_size=h_kernel, strides=(1, 1), decay=0.0001)

    ''' 3枚目(P) '''
    x3 = cba(input3, filters=16, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = cba(x3, filters=16, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = MaxPooling2D(strides=(2, 2), padding='same')(x3)

    x3 = cba(x3, filters=16, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = cba(x3, filters=16, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = MaxPooling2D(strides=(2, 2), padding='same')(x3)

    x3 = cba(x3, filters=32, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = cba(x3, filters=32, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = cba(x3, filters=32, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = MaxPooling2D(strides=(2, 2), padding='same')(x3)

    x3 = cba(x3, filters=32, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = cba(x3, filters=32, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = cba(x3, filters=32, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = MaxPooling2D(strides=(2, 2), padding='same')(x3)

    x3 = cba(x3, filters=64, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = cba(x3, filters=64, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)
    x3 = cba(x3, filters=64, kernel_size=p_kernel, strides=(1, 1), decay=0.0001)

    ''' ここから連結 '''
    merge = add([x1, x2, x3])

    x = GlobalAveragePooling2D(name='GAP')(merge)
    x = Dense(n_categories, kernel_initializer='he_normal', activation='softmax')(x)

    model = Model(inputs=[input1, input2, input3], outputs=x)

    return model


def wide_net(height, width, channel, n_categories):
    inputs = Input(shape=(height, width, channel))

    x_1 = cba(inputs, filters=16, kernel_size=(1, 2), strides=(1, 2), decay=0.0001)
    x_1 = cba(x_1, filters=16, kernel_size=(2, 1), strides=(2, 1), decay=0.0001)
    x_1 = cba(x_1, filters=32, kernel_size=(1, 2), strides=(1, 2), decay=0.0001)
    x_1 = cba(x_1, filters=32, kernel_size=(2, 1), strides=(2, 1), decay=0.0001)

    x_2 = cba(inputs, filters=16, kernel_size=(1, 4), strides=(1, 2), decay=0.0001)
    x_2 = cba(x_2, filters=16, kernel_size=(4, 1), strides=(2, 1), decay=0.0001)
    x_2 = cba(x_2, filters=32, kernel_size=(1, 4), strides=(1, 2), decay=0.0001)
    x_2 = cba(x_2, filters=32, kernel_size=(4, 1), strides=(2, 1), decay=0.0001)

    x_3 = cba(inputs, filters=16, kernel_size=(1, 6), strides=(1, 2), decay=0.0001)
    x_3 = cba(x_3, filters=16, kernel_size=(6, 1), strides=(2, 1), decay=0.0001)
    x_3 = cba(x_3, filters=32, kernel_size=(1, 6), strides=(1, 2), decay=0.0001)
    x_3 = cba(x_3, filters=32, kernel_size=(6, 1), strides=(2, 1), decay=0.0001)

    x_4 = cba(inputs, filters=16, kernel_size=(1, 8), strides=(1, 2), decay=0.0001)
    x_4 = cba(x_4, filters=16, kernel_size=(8, 1), strides=(2, 1), decay=0.0001)
    x_4 = cba(x_4, filters=32, kernel_size=(1, 8), strides=(1, 2), decay=0.0001)
    x_4 = cba(x_4, filters=32, kernel_size=(8, 1), strides=(2, 1), decay=0.0001)

    x = Add()([x_1, x_2, x_3, x_4])

    x = cba(x, filters=64, kernel_size=(1, 2), strides=(1, 2), decay=0.0001)
    x = cba(x, filters=64, kernel_size=(2, 1), strides=(2, 1), decay=0.0001)

    x = GlobalAveragePooling2D()(x)
    x = Dense(n_categories, kernel_initializer='he_normal', activation='softmax')(x)

    model = Model(inputs, x)

    return model


def resnet_custom(height, width, channel, n_categories):
    inputs = Input(shape=(height, width, channel))
    x = Convolution2D(16, (7, 7), strides=(1, 1), kernel_initializer='he_normal', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = _pre_act_resblock(n_filters=16)(x)
    x = _pre_act_resblock(n_filters=16)(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = _pre_act_resblock(n_filters=16)(x)
    x = _pre_act_resblock(n_filters=16)(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = _pre_act_resblock(n_filters=32)(x)
    x = _pre_act_resblock(n_filters=32)(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = _pre_act_resblock(n_filters=64)(x)
    x = _pre_act_resblock(n_filters=64)(x)

    # x = Flatten()(x)
    # x = Dense(1024, kernel_initializer='he_normal', activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1024, kernel_initializer='he_normal', activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_categories, kernel_initializer='he_normal', activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def resnet_34(height, width, channel, n_categories):
    inputs = Input(shape=(height, width, channel))
    x = Convolution2D(64, (7,7), strides=(1,1), kernel_initializer='he_normal', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)

    x = _resblock(n_filters=64)(x)
    x = _resblock(n_filters=64)(x)
    x = _resblock(n_filters=64)(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = _resblock(n_filters=128)(x)
    x = _resblock(n_filters=128)(x)
    x = _resblock(n_filters=128)(x)
    x = _resblock(n_filters=128)(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = _resblock(n_filters=256)(x)
    x = _resblock(n_filters=256)(x)
    x = _resblock(n_filters=256)(x)
    x = _resblock(n_filters=256)(x)
    x = _resblock(n_filters=256)(x)
    x = _resblock(n_filters=256)(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = _resblock(n_filters=512)(x)
    x = _resblock(n_filters=512)(x)
    x = _resblock(n_filters=512)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(n_categories, kernel_initializer='he_normal', activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model

