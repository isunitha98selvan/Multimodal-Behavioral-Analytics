import wave
import contextlib
import csv


data = []
finalData = []
def calculateDuration(fname):
	with contextlib.closing(wave.open('/home/rosageorge97/MajorProject/Audio/' + fname,'r')) as f:
		frames = f.getnframes()
		rate = f.getframerate()
		duration = frames / float(rate)
		return duration

def readCsv(fname):

	with open(fname, 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			temp = row
			temp[0] = temp[0].upper()
			temp[1] =  " ".join(temp[1].replace('Interviewer:','').replace('Interviewee:','').replace('Interviewer:','').replace('.','').replace('|','').split())

			data.append(temp)

def speechFeatures(id,text):
	
	duration = calculateDuration(id+'.wav')
	words = text.split()
	unique_words = set(text.split(' '))
	countWords = len(words)
	uniqueCountWords = len(unique_words)
	wps = countWords/duration
	uwps = uniqueCountWords/duration
	
	removeFillers = text.replace('like','').replace('uhh','').replace('um','').replace('umm','').replace('Umm','').replace('Mmm-hmm','').replace('hmm','').replace('Ahh','').replace('basically','')
	countremoveFillers = len(removeFillers.split())
	countFillers = countWords - countremoveFillers

	finalData.append([id, countWords, uniqueCountWords, wps, uwps, countFillers])

def main():     
	readCsv('/home/rosageorge97/MajorProject/Labels/transcripts.csv')
	for i in data:
		speechFeatures(i[0],i[1])
	print(finalData)
	with open('/home/rosageorge97/MajorProject/Results/result.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(finalData)
if __name__ == '__main__':
	main()