# python3 pyAudioAnalysis/audioAnalysis.py featureExtractionFile -i /home/sunitha/Documents/8th_sem/major_project/dataset/P1.wav -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050 -o output/
# python3 pyAudioAnalysis/audioAnalysis.py  featureExtractionDir -i /home/sunitha/Documents/8th_sem/major_project/dataset/ -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050
python3 pyAudioAnalysis/audioAnalysis.py  featureExtractionDir -i /home/rosageorge97/MajorProject/Audio/ -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050
python3 output/average_timeseries.py




