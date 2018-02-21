import csv
import numpy as np

class Node(object):

	def __init__(self):
		self.children = {}
		self.val = None
		self.criterion = None

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

	splitInfo = split_info(df_branches, df.shape[0])

	if splitInfo == 0:
		return 0
	else:
		return (info_d - info_b)/splitInfo

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

def majority_vote(df):
	y = list(df[:,0])
	return max([i for i in np.unique(y)], key=y.count)

def build_decision_tree(df, attribute_list, classes_):
	node = Node()
	y = df[:,0]
	if np.unique(y).size == 1:
		node.val = np.unique(y)[0]
		return node
	if not attribute_list:
		node.val = majority_vote(df)
		return node

	pattr = max(attribute_list, key=lambda x: gain_ratio(df,x))
	node.criterion = pattr
	attribute_list.pop(pattr-1)
	df_branches = split_df(df, pattr)
	class_ = classes_[pattr]

	for key in class_:
		if key not in df_branches:
			leaf = Node()
			leaf.val = majority_vote(df)
			node.children[key] = leaf
		else:
			node.children[key] = build_decision_tree(
													np.array(df_branches[key]),
													attribute_list,
													classes_)

	return node



def get_classes(df):
	X = list(df[:,1:])
	classes_ = [None]
	for i in zip(*X):
		classes_.append(list(np.unique(i)))

	return classes_

def main():
	## load data
	inputFile = 'mushroom.training.txt'
	df = load_data(inputFile)
	classes_ = get_classes(df)
	root = build_decision_tree(df, list(range(1,22)), classes_)
	print(root.children)
	
if __name__ == '__main__':
	main()