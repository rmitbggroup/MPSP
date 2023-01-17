import networkx as nx
import random
from node2vec import Node2Vec
import os
import numpy as np
import argparse
from gensim.models import KeyedVectors

def parse_args():
	parser = argparse.ArgumentParser(description="pretrain node embedding")

	parser.add_argument('--graph', nargs='?', default='BJ', help='input graph file')
	
	parser.add_argument('--dimensions', type=int, default='128',help='the number of embedding dimensions')

	return parser.parse_args()


		
def loadGraph(input):
	G=nx.Graph()

	with open(input) as f:	
		for line in f:
			strlist = line.split()
			if (len(strlist)>1):
				u=int(strlist[0])
				v=int(strlist[1])
				G.add_edge(u,v,weight=1)

				#G.add_edge(v,u)				


	return G

if __name__ == "__main__":
	args = parse_args()
	G=loadGraph(args.graph)
	node2vec = Node2Vec(G, dimensions=args.dimensions, workers=1,weight_key=None)
	model = node2vec.fit(min_count=1,workers=1)		
	outfile="embedding/"+args.graph
	model.wv.save(outfile)
	
