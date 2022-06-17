import numpy as np
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
import os

"""
References
    https://librosa.org/doc/main/auto_examples/plot_vocal_separation.html#sphx-glr-auto-examples-plot-vocal-separation-py
    https://librosa.org/doc/main/generated/librosa.feature.chroma_cens.html#librosa.feature.chroma_cens
"""

def extract_vocal(path, save_vocal=False, offset=None, duration=None):
    """
    parameters
        path: 음악 파일이 저장된 경로
        save_vocal : True이면 vocal을 추출한 부분의 wav 파일을 저장한다.
                    vocal이 잘 분리되었는지 확인할 수 있다.
        offset: vocal 추출을 시작할 부분(초)
        duration : vocal 추출할 부분의 duration
    returns
        y_foreground : vocal 부분만 추출된 wave
        sr : sampling rate
    """
    y, sr = librosa.load(path, offset=offset, duration=duration)

    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                margin_i * (S_full - S_filter),
                                power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                margin_v * S_filter,
                                power=power)

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    y_foreground = librosa.istft(S_foreground * phase)

    if save_vocal:
        title = Path(path).stem
        pardir = os.path.join(path, '..')
        sf.write(os.path.join(pardir,title+"_vocal.wav"), y_foreground, sr, subtype='PCM_24')

    return y_foreground, sr




if __name__=="__main__":
    from tkinter import filedialog
    import pandas as pd
    df = pd.read_csv('./music/song_list.csv')
    files = filedialog.askopenfilenames()
    

    for file in files:
        # csv파일에서 이미 정의된 부분
        song_title = Path(file).stem # 음악의 제목
        offset = df.loc[song_title, "offset"] # 몇초부터
        duration = df.loc[song_title, "duration"] # 몇초동안의 음원을 분리할 것인지 설정

        vocal, sr = extract_vocal(file,save_vocal=True, offset=offset, duration=duration) 
        chroma_cens = librosa.feature.chroma_cens(vocal, sr)

        pardir = os.path.join(file,'..')
        npy_path = os.path.join(pardir, song_title)+'.npy'
        with open(npy_path, 'wb') as f:
            np.save(f, chroma_cens)
