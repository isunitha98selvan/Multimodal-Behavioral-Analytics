import pandas as pd
import csv
import urllib
import glob,os
import parselmouth
import numpy as np 
import matplotlib.pyplot as plt

def pitch_values(snd):
    # calculate the min,max,mean,std.dev and abs pitch
    pitch = snd.to_pitch()
    duration = snd.get_total_duration()
    mean_pitch = parselmouth.praat.call(pitch, "Get mean",0,duration,"Hertz")
    min_pitch = parselmouth.praat.call(pitch, "Get minimum",0,duration,"Hertz","Parabolic")
    max_pitch = parselmouth.praat.call(pitch, "Get maximum",0,duration,"Hertz","Parabolic")
    
    # print(mean_pitch,min_pitch,max_pitch)
    return duration,mean_pitch,min_pitch,max_pitch

def get_base_features(snd):
    power = snd.get_power()
    intensity = snd.get_intensity()
    return power,intensity

def get_spectrogram(snd, dynamic_range = 70):
    spectrogram = snd.to_spectrogram()  
    plt.figure()
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    np.seterr(divide = 'ignore')
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")
    # print(spectrogram)
    return spectrogram

def convert_csv(filename):
    # 1 - Zero crossing rate
    # 2 - Energy
    # 3 - Entropy of Energy
    # 4 - Spectral Centroid
    # 5 - Spectral Spread
    # 6 - Spectral Entropy
    # 7 - Spectral Flux
    # 8 - Spectral Rolloff
    # 9-21 - MFCCs
    # 22-33 - Chroma vector 
    # 34 - Chroma Deviation

    df = pd.read_csv(filename, header=None)
    rows = len(df)
    avg_values = df.sum(axis = 0, skipna = True)/rows
    feature_vector = avg_values.values.tolist()
    return feature_vector

def main():
    audio_files = []
    path = '/home/rosageorge97/MajorProject/Audio/'
    # path = "/home/sunitha/Documents/8th_sem/major_project/dataset/"
    for filename in glob.glob(os.path.join(path, '*.wav')):
        audio_files.append(filename)

    print(audio_files)
    i=1
    for file in audio_files:
        snd = parselmouth.Sound(file)
        power,intensity = get_base_features(snd)
        duration,mean_pitch,min_pitch,max_pitch = pitch_values(snd)
        spectrogram = get_spectrogram(snd)
        # print(file)
        end_name = file.rsplit('/', 1)[-1]
        csv_file = path+ end_name+"_st.csv"
        audio_analysis = convert_csv(csv_file)
        feature_vector= [end_name,power,intensity,duration,mean_pitch,min_pitch,max_pitch]
        for value in audio_analysis:
            feature_vector.append(value)
        i+=1
        with open('/home/rosageorge97/MajorProject/Results/audio_features.csv', 'a', newline='') as file:
        # with open('audio_features.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(feature_vector)
        
        # print(feature_vector)
        # print(len(feature_vector))

if __name__ == "__main__":
    main()
