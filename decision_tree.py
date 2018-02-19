import csv

## load data

def main():
	reader = csv.reader(open('mushroom.training.txt'), delimiter='\t')
	df = [row for row in reader]
	
if __name__ == '__main__':
	main()