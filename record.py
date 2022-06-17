import pyaudio
import wave
import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
import glob
from pathlib import Path
import pandas as pd

"""
References
    http://people.csail.mit.edu/hubert/pyaudio/
    https://librosa.org/doc/main/generated/librosa.yin.html
    https://stackoverflow.com/questions/59686267/tkinter-start-stop-button-for-recording-audio-in-python
    https://stackoverflow.com/questions/43521804/recording-audio-with-pyaudio-on-a-button-click-and-stop-recording-on-another-but
"""

RECORD_SECONDS = 5

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "output.wav"


######################## 녹음 ##################################
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording *")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* stop recording *")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

y, sr = librosa.load(WAVE_OUTPUT_FILENAME, sr=RATE)
X = librosa.feature.chroma_cens(y=y, sr=sr)

# 사용자가 녹음한 음성의 chroma feature
librosa.display.specshow(X, y_axis="chroma",x_axis='time')
plt.title("Chroma CENS feature of recorded audio"),plt.show()


#############################################################################
###### 녹음된 음성과 저장된 vocal 음원들 비교 & 최소 DTW cost 갖는 곡 선택 ######
#############################################################################
df = pd.read_csv('./music/song_list.csv')

costs = dict() # 녹음한 음성과 저장된 vocal 음원들의 cost 저장
files = glob.glob('./music/*.npy') # vocal 음원들에 대한 chroma feature
for file in files:
    song_title = Path(file).stem

    Y = np.load(file) # 저장된 음원의 chroma feature
    D, wp = librosa.sequence.dtw(X, Y, subseq=True) # DTW matrix

    # 저장된 보컬 음원의 chroma feature
    librosa.display.specshow(Y, y_axis="chroma",x_axis='time')
    plt.title(f"Chroma CENS feature of {song_title}")

    fig, ax = plt.subplots(nrows=2, sharex=True)

    # 사용자 음성과 vocal 음원의 DTW, optimal path 
    img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                                ax=ax[0])
    ax[0].set(title=f'DTW cost with {song_title}',
                xlabel='Noisy sequence', ylabel='Target')
    ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax[0].legend()
    fig.colorbar(img, ax=ax[0])

    # Mathcing cost function
    ax[1].plot(D[-1, :] / wp.shape[0])
    ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2],
            title='Matching cost function')
    plt.show()

    N, M = D.shape

    cost = D[N-1, M-1]
    duration = df.loc[song_title, "duration"]
    cost /= duration # 시퀀스 길이 차이에 대해 보정
    costs[song_title] = cost
    print(f"title: {song_title}, accumulated cost:{cost}")

min_cost = min(costs.values()) # 가장 작은 cost 값
for title, cost in costs.items():
    # cost가 가장 작은 곡을 선택
    if min_cost == cost:
        print(f"The song that you're finding is '{title}'")