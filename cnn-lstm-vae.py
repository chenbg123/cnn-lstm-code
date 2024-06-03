import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, RepeatVector, TimeDistributed, \
    Reshape
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# 生成示例多特征数据
def generate_sample_multivariate_data(num_features=3):
    time = np.arange(0, 100, 0.1)
    data = np.array([np.sin(time + i) + np.random.normal(0, 0.1, len(time)) for i in range(num_features)]).T
    return data


# 准备数据
def prepare_multivariate_data(series, n_steps):
    X, y = [], []
    for i in range(len(series)):
        end_ix = i + n_steps
        if end_ix > len(series) - 1:
            break
        seq_x, seq_y = series[i:end_ix, :], series[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# 构建CNN-LSTM-VAE模型
def build_multivariate_model(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu')(x)

    # LSTM Encoder
    x = RepeatVector(input_shape[0])(x)
    x = LSTM(100, activation='relu', return_sequences=True)(x)

    # Decoder
    x = LSTM(100, activation='relu', return_sequences=True)(x)
    x = TimeDistributed(Dense(input_shape[1]))(x)

    # Output
    x = TimeDistributed(Dense(input_shape[1]))(x)
    x = Flatten()(x)
    outputs = Dense(input_shape[1])(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


# 主函数
def main():
    # 生成示例多特征数据
    num_features = 3
    data = generate_sample_multivariate_data(num_features)

    # 标准化数据
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # 准备训练和测试数据
    n_steps = 10
    X, y = prepare_multivariate_data(data, n_steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建模型
    model = build_multivariate_model((n_steps, num_features))

    # 训练模型
    model.fit(X_train, y_train, epochs=20, verbose=1, validation_data=(X_test, y_test))

    # 预测
    y_pred = model.predict(X_test)

    # 反标准化预测值
    y_test = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred)

    # 绘制结果
    plt.figure(figsize=(15, 8))
    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.plot(y_test[:, i], label='True Feature {}'.format(i))
        plt.plot(y_pred[:, i], label='Predicted Feature {}'.format(i))
        plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
