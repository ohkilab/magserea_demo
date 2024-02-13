import multiprocessing
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import deque
from itertools import islice
import tkinter as tk
import tkinter.ttk as ttk
import math
from scipy.signal import argrelmax,argrelmin
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from tensorflow.python.keras.models import load_model
from tensorflow import keras
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

from sklearn.metrics import mean_squared_error
import sys



def sensor_data(sensor_display_queue, sensor_peak_queue, filename):

        
    df = pd.read_csv("./demo_data/" + filename + ".csv", header=None)
    i = 0
    alpha = 0.8

    pre_x_mag = 0.0
    pre_y_mag = 0.0
    pre_z_mag = 0.0
    pre_x_gyro = 0.0
    pre_y_gyro = 0.0
    pre_z_gyro = 0.0

    while True:

        #センサーデータ受信
        x_mag = df.loc[i][3] / 120
        y_mag = df.loc[i][4] / 120
        z_mag = df.loc[i][5] / 120
        x_gyro = df.loc[i][0]
        y_gyro = df.loc[i][1]
        z_gyro = df.loc[i][2]

        #low_pass_filter
        x_mag = alpha * pre_x_mag + (1 - alpha) * x_mag
        pre_x_mag = x_mag
        y_mag = alpha * pre_y_mag + (1 - alpha) * y_mag
        pre_y_mag = y_mag
        z_mag = alpha * pre_z_mag + (1 - alpha) * z_mag
        pre_z_mag = z_mag
        x_gyro = alpha * pre_x_gyro + (1 - alpha) * x_gyro
        pre_x_gyro = x_gyro
        y_gyro= alpha * pre_y_gyro + (1 - alpha) * y_gyro
        pre_y_gyro = y_gyro
        z_gyro= alpha * pre_z_gyro + (1 - alpha) * z_gyro
        pre_z_gyro = z_gyro

        #peak検知に使用する角速度データ
        norm_gyro = y_gyro + z_gyro
        
        sensor_datas = [x_mag, y_mag, z_mag, norm_gyro]
        sensor_display_queue.put(sensor_datas)  
        sensor_peak_queue.put(sensor_datas)
        time.sleep(0.02)  # 0.02秒待つ

        i += 1



# sensorデータ表示関数
def data_display(sensor_display_queue, data_display_and_sensor_peak_stop_init_event):
    global xs, x, x_mag, y_mag, z_mag, norm_gyro

    def init_data():
        global xs, x, x_mag, y_mag, z_mag, norm_gyro        
        #x軸の値
        xs = deque()
        x = 0
        #実際のデータ
        x_mag = deque([])
        y_mag = deque([])
        z_mag = deque([])
        norm_gyro = deque([])


    init_data()

    #グラフ関係
    fig = plt.figure(figsize = (10,6))
    fig.suptitle('sensor data')
    #グラフを描画するsubplot領域を作成。
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.set_ylabel("magnetic field")
    ax2.set_ylabel("angular velocity")

    while True:
        # キューからデータを取得
        sensor_data = sensor_display_queue.get()

        if data_display_and_sensor_peak_stop_init_event.is_set():
            init_data()
    
        else:

            sensor_x_mag, sensor_y_mag, sensor_z_mag, sensor_norm_gyro = sensor_data

            #値を追加
            x_mag.append(sensor_x_mag)
            y_mag.append(sensor_y_mag)
            z_mag.append(sensor_z_mag)
            norm_gyro.append(sensor_norm_gyro)

            #最大範囲
            if x < 300:
                xs.append(x)
                #値を更新
                x += 1
            else:
                #先頭要素を削除
                x_mag.popleft()
                y_mag.popleft()
                z_mag.popleft()
                norm_gyro.popleft()

                
                
            #plot
            line1, = ax1.plot(xs, x_mag, color="steelblue", label="x")
            line2, = ax1.plot(xs, y_mag, color="olivedrab", label="y")
            line3, = ax1.plot(xs, z_mag, color="tomato", label="z")

            line4, = ax2.plot(xs, norm_gyro, color="mediumpurple")

            ax1.legend(loc='upper left')

            # 0.001秒停止
            plt.pause(0.001)
            #描画したグラフを削除
            line1.remove()
            line2.remove()
            line3.remove()
            line4.remove()

        


#peak検知間数
def peak_detection(sensor_peak_queue, peak_detection_start_event, peak_display_queue, data_display_and_sensor_peak_stop_init_event):
    #右側押し
    def peak_cut_data(data):
        is_sucsess = True
        peak_cut_data = data[:, 0:3]

        norm_gyro = data[:, 3]
        #前後20データと比較してピークなindexを取得
        maxs = argrelmax(norm_gyro, order=20)
    
        #ピークかつ，値が50以上のもののみを抽出
        max_peak = 0
        for j in range(len(maxs[0])):
            if norm_gyro[maxs[0][j]] > 50 :
                max_peak = maxs[0][j]

        mins = argrelmin(norm_gyro, order=20)
        #マイナスのピークかつ値が-50以下かつmax_peakから20データ離れている
        min_peak = 0
        for j in range(len(mins[0])):
            if norm_gyro[mins[0][j]] < -50 and max_peak + 20 < mins[0][j] :
                min_peak = mins[0][j]
        
        #ピークを見つけることができなかった
        if min_peak == 0 or max_peak ==0 :
            is_sucsess = False

        if is_sucsess:
            peak_cut_data = data[max_peak:min_peak, 0:3]
        return is_sucsess, peak_cut_data
    

    WINDOW_SIZE = 300
    buf = deque([])
    # time.sleep(0.5)
    while True:
        #センサデータを取得
        sensor_data = sensor_peak_queue.get()

        #ピーク検知により対象区間が抽出されたとき
        if data_display_and_sensor_peak_stop_init_event.is_set():
            buf = deque([])
        else:
            # time.sleep(0.02)

            #センサデータはbufに追加していく
            buf.append(sensor_data)
            
            if len(buf) >= WINDOW_SIZE + 1:
                
                window_data = np.array(list(islice(buf, 0, WINDOW_SIZE)))

                #GUIでpeak検知開始ボタンが押された
                if peak_detection_start_event.is_set():
                    is_sucsess, peak_cut_data = peak_cut_data(window_data)

                    #peak検知できた
                    if is_sucsess :
                        #ピーク検知開始ボタンをクリア
                        peak_detection_start_event.clear()
                        peak_display_queue.put(peak_cut_data)
                

                buf.popleft()
    

#ピーク検知により抽出されたデータの表示
def peak_cut_data_display(peak_display_queue, data_display_and_sensor_peak_stop_init_event, peak_cut_lstm_queue):
    
    while True:

        sensor_data = peak_display_queue.get()
        #センサデータの表示とピーク検知をストップする
        data_display_and_sensor_peak_stop_init_event.set()

        pd_data = pd.DataFrame(sensor_data, range(sensor_data.shape[0]), range(sensor_data.shape[1]))
        pd_data.columns = ['x','y','z']
        x = list(range(sensor_data.shape[0]))

        fig = make_subplots(rows=1, cols=3, subplot_titles=['x', 'y', 'z'])
        fig.add_trace(go.Scatter(x=x, y=pd_data['x'], mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=pd_data['y'], mode='lines'), row=1, col=2)
        fig.add_trace(go.Scatter(x=x, y=pd_data['z'], mode='lines'), row=1, col=3)

        fig.update_layout(yaxis_title='magnetic field', title_text='', showlegend=False)
        fig.show()

        #lstmに送信
        peak_cut_lstm_queue.put(sensor_data)
  
#lstmに入力し，結果を表示
def lstm_result(finish_event, peak_cut_lstm_queue, select_adopt_event, select_reject_event):
    #前処理標準化
    def preprocess_data(data):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        return data, scaler

    def create_dataset_pred_length(dataset, look_back, pred_length):
        dataX, dataY = [], []
        #+1する理由は，範囲の添字的に，+1の値が未満を表しているから
        for i in range(len(dataset) - look_back - pred_length + 1):
            a = dataset[i:(i+look_back), :]
            b = dataset[(i + look_back) : (i + look_back + pred_length), :]
            dataX.append(a)
            dataY.append(b)

        return np.array(dataX), np.array(dataY)

    def model_score_pred_length(model, testX, testY, pred_length):
        testPredict = model.predict(testX)
        testScore = mean_squared_error(testY.reshape(testY.shape[0], pred_length * 3), testPredict)
        return testScore

    while True:
        isAdopt = False
        data = peak_cut_lstm_queue.get()
        break
        while True:
            #採用
            if select_adopt_event.is_set():
                select_adopt_event.clear()
                isAdopt = True
                break
            #不採用
            if select_reject_event.is_set():
                select_reject_event.clear()
                break

        if not isAdopt:
            continue

        if isAdopt:
            break

    finish_event.set()

    #学習済みモデル
    model = load_model('./model-100-0.0269.h5')
    #EERth
    threshold = -0.06362345769103415

    look_back = 25
    test, _ = preprocess_data(data)
    
    testX, testY = create_dataset_pred_length(test, look_back, 10)
    predict = model.predict(testX)
    score = model_score_pred_length(model, testX, testY, 10)
    score = score * -1


    matplotlib.use('TkAgg')
    x = list(range(test.shape[0]))
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))

    testY = testY.reshape(testY.shape[0], 10 * 3)

    for i in range(testY.shape[0]):
        x = []
        y = []
        y1 = []
        y2 = []
        for j in range(10):
            x.append(i + j)
            y.append(testY[i, j * 3])
            y1.append(testY[i, j * 3 + 1])
            y2.append(testY[i, j * 3 + 2])
        ax[0, 0].plot(x, y)
        ax[0, 1].plot(x, y1)
        ax[0, 2].plot(x, y2)

    ax[0, 0].set_title('input x')
    ax[0, 1].set_title('input y')
    ax[0, 2].set_title('input z')

    for i in range(predict.shape[0]):
        x = []
        y = []
        y1 = []
        y2 = []
        for j in range(10):
            x.append(i + j)
            y.append(predict[i, j * 3])
            y1.append(predict[i, j * 3 + 1])
            y2.append(predict[i, j * 3 + 2])
        ax[1, 0].plot(x,y)
        ax[1, 1].plot(x, y1)
        ax[1, 2].plot(x, y2)

    ax[1, 0].set_title('output x')
    ax[1, 1].set_title('output y')
    ax[1, 2].set_title('output z')


    plt.tight_layout()

    root = tk.Tk()
    # root.geometry("2000x1500")
    # MatplotlibのフィギュアをTkinterのキャンバスに埋め込む
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    # キャンバスを配置
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    gosa_labe = tk.Label(root, text="Reconstruction Error: " + f"{score:.4f}", font=("Helvetica", 24))
    gosa_labe.pack()
    th_labe = tk.Label(root, text="EER Threshold: " + f"{threshold:.4f}", font=("Helvetica", 24))
    th_labe.pack()

    if score > threshold :
        ok_label = tk.Label(root, text="Accept", font=("Helvetica", 36), fg="#dc143c")
        ok_label.pack()
    else:
        ng_label = tk.Label(root, text="Reject", font=("Helvetica", 36), fg="#4169e1")
        ng_label.pack()


    # Tkinterメインループの開始
    root.mainloop()

#userinterctionによる動作の制御
def user_interaction(peak_detection_start_event, select_adopt_event, select_reject_event, data_display_and_sensor_peak_stop_init_event):
    filename = "./real_time_system_userInteraction_data.txt"
    
    pre_interaction_data = np.loadtxt(filename)
    while True:

        
        interaction_data = np.loadtxt(filename)

        #peak検知開始ボタンが押された
        if pre_interaction_data[0] != interaction_data[0]:
            print("0")
            peak_detection_start_event.set()

        if pre_interaction_data[1] != interaction_data[1]:
            print("1")
            select_adopt_event.set()
        if pre_interaction_data[2] != interaction_data[2]:
            print("2")
            #データの再取得とピーク検知のためにデータの読み込み開始＆不採用を通知
            data_display_and_sensor_peak_stop_init_event.clear()
            select_reject_event.set()

        pre_interaction_data = interaction_data

        time.sleep(0.1)  # 0.1秒待つ


if __name__ == "__main__":

    argv = sys.argv
    filename = argv[1]

    #queue
    sensor_display_queue = multiprocessing.Queue()  # マルチプロセス対応のキューを生成
    sensor_peak_queue = multiprocessing.Queue()
    peak_display_queue = multiprocessing.Queue()
    peak_cut_lstm_queue = multiprocessing.Queue()
    
    #event
    peak_detection_start_event = multiprocessing.Event()
    data_display_and_sensor_peak_stop_init_event = multiprocessing.Event()
    select_adopt_event = multiprocessing.Event()
    select_reject_event = multiprocessing.Event()
    finish_event = multiprocessing.Event()

    # プロセスを生成
    sensor_data_proc = multiprocessing.Process(target=sensor_data, args=(sensor_display_queue, sensor_peak_queue, filename))
    data_display_proc = multiprocessing.Process(target=data_display, args=(sensor_display_queue, data_display_and_sensor_peak_stop_init_event))
    sensor_peak_proc = multiprocessing.Process(target=peak_detection, args=(sensor_peak_queue, peak_detection_start_event, peak_display_queue, data_display_and_sensor_peak_stop_init_event))
    peak_cut_data_display_proc = multiprocessing.Process(target=peak_cut_data_display, args=(peak_display_queue, data_display_and_sensor_peak_stop_init_event, peak_cut_lstm_queue))
    user_interaction_proc = multiprocessing.Process(target=user_interaction, args=(peak_detection_start_event, select_adopt_event, select_reject_event, data_display_and_sensor_peak_stop_init_event))
    lstm_result_proc = multiprocessing.Process(target=lstm_result, args=(finish_event, peak_cut_lstm_queue, select_adopt_event, select_reject_event))

    # プロセスを開始
    sensor_data_proc.start()
    data_display_proc.start()
    sensor_peak_proc.start()
    user_interaction_proc.start()
    peak_cut_data_display_proc.start()
    lstm_result_proc.start()

    #autoencoder以外終了
    while True:
        if finish_event.is_set():
            sensor_data_proc.terminate()
            data_display_proc.terminate()
            sensor_peak_proc.terminate()
            user_interaction_proc.terminate()
            peak_cut_data_display_proc.terminate()
        
            break

    sensor_data_proc.join()
    data_display_proc.join()
    sensor_peak_proc.join()
    user_interaction_proc.join()
    peak_cut_data_display_proc.join()
    lstm_result_proc.join()