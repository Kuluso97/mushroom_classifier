import csv
import numpy as np

def load_data(inputFile):
	reader = csv.reader(open(inputFile), delimiter='\t')
	df = np.array([row for row in reader])
	return df

def gain_ratio(df, attrIndex):
	info_d = get_info(df)
	df_branches = split_df(df, attrIndex)
	info_b = 0
	for key in df_branches:
		freq = len(df_branches[key])*1.0/df.shape[0]
		info_b += freq * get_info(np.array(df_branches[key]))

	return (info_d - info_b)/split_info(df_branches, df.shape[0])

def split_df(df, attrIndex):
	classes_ = np.unique(df[:, attrIndex])
	df_branches = {class_: [] for class_ in classes_}

	for row in df:
		class_ = row[attrIndex]
		df_branches[class_].append(row)

	return df_branches

def get_info(df):
	y = list(df[:,0])
	classes_ = np.unique(y)
	probs = [y.count(class_) * 1.0 / len(y) for class_ in classes_]
	return - sum([pi * np.log2(pi) for pi in probs])

def split_info(df_branches, total):
	res = 0
	for key in df_branches:
		freq = len(df_branches[key])*1.0/total
		res += - freq * np.log2(freq)

	return res

def main():
	## load data
	inputFile = 'mushroom.training.txt'
	df = load_data(inputFile)
	print(gain_ratio(df, 2))	
	
if __name__ == '__main__':
	main()