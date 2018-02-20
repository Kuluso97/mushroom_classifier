import csv
import numpy as np

def load_data(inputFile):
	reader = csv.reader(open(inputFile), delimiter='\t')
	df = np.array([row for row in reader])
	return df[:,1:], df[:,0]

def main():
	## load data
	inputFile = 'mushroom.training.txt'
	X, y = load_data(inputFile)	
	
if __name__ == '__main__':
	main()