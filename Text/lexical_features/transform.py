import csv

data = []
finalData = []
finalData.append(['Id','Joy', 'Sadness', 'Tentative', 'Analytical', 'Fear', 'Anger'])
def readCsv(fname):

	with open(fname, 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			data.append(row[1:])
			finalData.append([row[0]])
def Convert(lst): 
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)} 
    return res_dct

def makeFinalTable():
	ptr = 1
	for i in data:
		temp = Convert(i)
		finalData[ptr].append(temp.get('Joy'))
		finalData[ptr].append(temp.get('Sadness'))
		finalData[ptr].append(temp.get('Tentative'))
		finalData[ptr].append(temp.get('Analytical'))
		finalData[ptr].append(temp.get('Fear'))
		finalData[ptr].append(temp.get('Anger'))
		ptr+=1
		

readCsv('ToneAnalyzer.csv')

makeFinalTable()

with open('ToneAnalyzer_transformed.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(finalData)
		