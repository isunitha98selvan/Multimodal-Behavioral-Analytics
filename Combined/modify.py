import csv
data = []

with open('audio_features.csv', 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			data.append(row)
for i in data:
	i[0] = i[0].split('.')[0]
	
with open('audio_features_final.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	writer.writerows(data)
