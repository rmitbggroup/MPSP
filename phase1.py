import sys
import random
import networkx as nx
import argparse
import heapq
import time
import csv
from collections import deque 
#from numpy.random import choice


def parse_args():
	parser = argparse.ArgumentParser(description="phase 1: candidate paths generation")

	parser.add_argument('--graph', nargs='?', default='BJ', help='input graph file')

	parser.add_argument('--output', nargs='?', default='validation', help='input graph file')

	parser.add_argument('--m', type=int, default='20',help='the number of candidate paths')

	parser.add_argument('--pairs', type=int, default='100',help='the number of s-t pairs')


	return parser.parse_args()



class MPSP(object):
	"""docstring for MPSP"""
	

	def __init__(self, args):
		super(MPSP, self).__init__()
		
		self.m=args.m
		self.input=args.graph	
		self.output=args.output+"/"+self.input
		self.G = self.loadGraph()
		self.pairs=self.getPairs(args.pairs)


	def getPairs(self,num):
		pairs=[]
		count=0

		while(count<num):

			pair=(random.choice(self.G.nodes()),random.choice(self.G.nodes()))
			
			if (pair[0]==pair[1]):
				continue
			#pair=(random.randint(0,len(self.G)),random.randint(0,len(self.G)))
			pairs.append(pair)
			count+=1



		return pairs
		
	def loadGraph(self):
		G=nx.Graph()
		with open(self.input) as f:	
			for line in f:
				strlist = line.split()

				u=int(strlist[0])
				v=int(strlist[1])
				weight=float(strlist[2])
				prob=float(strlist[3])
				G.add_edge(u,v,weight=weight,prob=prob)		

					#G.add_edge(v,u)	


		return G

	def generatePath(self,source,dest):
		for u in self.G.nodes():
			self.G.node[u]['distance']=sys.maxsize

		self.G.node[source]['distance'] = 0

		parents={source:-1}
		pq = [(0, source,-1)]
		while (pq):
			
			current_dis, current_vertex,parent = heapq.heappop(pq)

			# Nodes can get added to the priority queue multiple times. We only
			# process a vertex the first time we remove it from the priority queue.
			if current_dis > self.G.node[current_vertex]['distance']:
				continue

			parents[current_vertex]=parent

			if (current_vertex==dest):
				break

			for v in self.G.neighbors(current_vertex):
				dis=current_dis+self.G[current_vertex][v]['weight']
				if (dis < self.G.node[v]['distance'] and random.random()<=self.G[current_vertex][v]['prob']):
					self.G.node[v]['distance']=dis
					heapq.heappush(pq, (dis, v,current_vertex))

		return parents



	def getPath(self,dest,parents):


		path=[]
		start=dest
		while (start!=-1):
			path.append(start)
			start=parents[start]

		path.reverse()


		return tuple(path)

	def writePaths(self):
		with open(self.output,"a") as f:
			for source,dest in self.pairs:
				
				start=time.time()
				candidates=set()
				for i in range(self.m):
					parents=self.generatePath(source,dest)
					if (dest in parents):
						path=self.getPath(dest,parents)
						candidates.add(path)
				phase1=time.time()-start

				if (not candidates):
					continue
				
				f.write("#\n")
				f.write(str(phase1)+"\n")
				for candidate in candidates:
					
					for node in candidate:
						f.write(str(node)+" ")
					f.write("\n")
				f.write("\n")

		
if __name__ == "__main__":
	args = parse_args()
	mpsp=MPSP(args)
	mpsp.writePaths()