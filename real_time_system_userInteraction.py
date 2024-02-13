import tkinter as tk
import tkinter.ttk as ttk

import numpy as np
import pandas as pd
import os
import time

#押されたボタンに対応して値を保存
def clicked_peak_detection_start_button():
    interaction_data[0] = not interaction_data[0]
    np.savetxt(filename, interaction_data, fmt='%d')
def clicked_adopt_button():
    interaction_data[1] = not interaction_data[1]
    np.savetxt(filename, interaction_data, fmt='%d')
def clicked_reject_button():
    interaction_data[2] = not interaction_data[2]
    np.savetxt(filename, interaction_data, fmt='%d')


filename = "./real_time_system_userInteraction_data.txt"
#ファイルが存在しない場合は作成
if not os.path.exists(filename):
    np.savetxt(filename, np.array([False, False, False]), fmt='%d')
#ファイルの読み込み
interaction_data = np.loadtxt(filename)

#GUI
root = tk.Tk()
root.title("action")
root.geometry("200x70")

peak_label = tk.Label(root, text="ピーク検知")
peak_detection_start_button = tk.Button(root, text="開始", command=clicked_peak_detection_start_button)
peak_label.grid(row=1, column=1)
peak_detection_start_button.grid(row=1, column=2)


root.mainloop()
