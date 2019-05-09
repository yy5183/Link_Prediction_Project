import networkx as nx
import pandas as pd

class LinkRecommendation():
	trainG = None
	testG = None
	k = None   # top k recommendation
	sim = None  # similarity matrix

	def __init__(self, dataPath, k):
		self.k = k
		df = pd.read_csv(dataPath, sep=' ', header=None)
		test_df = df.sample(frac=0.2, random_state=10)
		train_df = df.drop(test_df.index)

		self.trainG = nx.Graph()
		self.testG = nx.Graph()

		train_data = train_df.values
		test_data = test_df.values

		for i in range(len(train_data)):
			self.trainG.add_edge(train_data[i][0], train_data[i][1])

		for i in range(len(test_data)):
			self.testG.add_edge(test_data[i][0], test_data[i][1])

	def katz(self):
		# code here
		return None

	def xgbst(self):
		# code here
		return None

	def local_methods(self, method):
		train_n = len(self.trainG.nodes)
		nodes = list(self.trainG.nodes)

		sim = [[0 for i in range(train_n)] for j in range(train_n)]
		for i in range(train_n):
			for j in range(train_n):
				if i != j and i in self.trainG.nodes and j in self.trainG.nodes:
					sim[i][j] = self.local_methods_calSimilarity(nodes[i], nodes[j], method)

		self.sim = sim
		self.precision_and_recall()

	def local_methods_calSimilarity(self, i, j, method):
		if method == "CN":
			return len(set(self.trainG.neighbors(i)).intersection(set(self.trainG.neighbors(j))))

		elif method == "JA":
			# code here
			return None

		elif method == "AA":
			# code here
			return None

		elif method == "PA":
			# code here
			return None

	def precision_and_recall(self):
		precision = recall = c = 0
		for person in self.testG.nodes:
			if person in self.trainG.nodes:
				testNeighbors = [n for n in self.testG.neighbors(person)]
				if len(testNeighbors) < self.k:
					self.k = len(testNeighbors)
				top_k = set(self.top_k_rec(person))
				precision += len(top_k.intersection(testNeighbors)) / float(self.k)
				recall += len(top_k.intersection(testNeighbors)) / float(len(testNeighbors))
				c += 1
		print("Precision is : " + str(precision / c))
		print("Recall is : " + str(recall / c))


	def top_k_rec(self, person):
		nodes = list(self.trainG.nodes)
		indexPerson = nodes.index(person)
		indexRecs = sorted(filter(lambda indexX: indexPerson!=indexX and not self.trainG.has_edge(person,nodes[indexX]),
								  range(len(self.sim[indexPerson]))),
					   key=lambda indexX: self.sim[indexPerson][indexX],reverse=True)[0:self.k]
		recFriends = [nodes[i] for i in indexRecs]
		return recFriends

	def build_model(self, method):
		if method == "CN" or method == "JA" or method == "AA" or method == "PA":
			self.local_methods(method)
		elif method == "katz":
			self.katz()
		elif method == "xgboost":
			self.xgbst()


# Local methods: CN(common_neighbors) JA(jaccard) AA(adamic_adar) PA(preferential_attachment)
# katz
# xgboost

dataPath = 'data/test.txt'
# dataPath = 'data/facebook_combined.txt'
LP = LinkRecommendation(dataPath,5)
LP.build_model('CN')
print(LP.top_k_rec(0))


