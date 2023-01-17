import sys
import random
import networkx as nx
import argparse
import heapq
import time
import csv
import tensorflow as tf
import logging
import numpy as np
from gensim.models import KeyedVectors
from node2vec.edges import HadamardEmbedder
import os
import numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def parse_args():
	parser = argparse.ArgumentParser(description="phase 2: return MPSPs")

	parser.add_argument('--trainedGraph', nargs='?', default='BJ', help='the graph the model is trained in')

	parser.add_argument('--targetGraph', nargs='?', default='BJ', help='the graph the trained model is applied in')

	parser.add_argument('--N', type=int, default='1000',help='the number of monte carlo simulation')

	parser.add_argument('--dimensions', type=int, default='128',help='the dimension size')

	parser.add_argument('--counter', type=int, default='500',help='the number of test pairs')

	parser.add_argument('--size', type=int, default='20',help='the number of candidate paths')

	parser.add_argument('--m', type=int, default='20',help='the number of random walk')

	parser.add_argument('--ratio', type=float, default='0.1',help='the percentage of graph size for sampling starting nodes')

	parser.add_argument('--length', type=int, default='30',help='the length of walk')

	parser.add_argument('--embedder', nargs='?', default='L1', help='Hadamard, Mean, L1 and L2')


	return parser.parse_args()



class MPSP(object):
	"""docstring for MPSP"""
	
	def __init__(self, args):
		super(MPSP, self).__init__()
		self.suffix="_op"+str(args.embedder)+"_m"+str(args.m)+"_l"+str(args.length)+"_r"+str(args.ratio)


		self.targetGraph=args.targetGraph
		self.trainedGraph=args.trainedGraph	
		if (args.targetGraph=="None"):
			self.targetGraph=self.trainedGraph
		
		self.N=args.N	
		self.counter=args.counter
		self.kvalues=[1,5,10]
		self.kcounters={}
		for value in self.kvalues:
			self.kcounters[value]=self.counter
		self.size=args.size
		self.paths="test/"+self.targetGraph
		self.embedder=args.embedder
		self.embedding="embedding/"+self.targetGraph
		self.G = self.loadGraph()
		self.dimensions=args.dimensions	
		tf.reset_default_graph()
		self.embholder=tf.placeholder(dtype=tf.float32,shape=(None,None))
		self.mlp1=tf.get_variable("mlp1",shape=[self.dimensions*2, self.dimensions])
		self.mlp2=tf.get_variable("mlp2",shape=[self.dimensions, 1])

		self._predictC()
		saver = tf.train.Saver()

		config = tf.ConfigProto()

		os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
		self.sess=tf.Session(config=config)
		saver.restore(self.sess, "models/"+self.trainedGraph+self.suffix+"/model")	

	def loadGraph(self):
		G=nx.Graph()


		with open(self.targetGraph) as f:	
			for line in f:
				strlist = line.split()

				u=int(strlist[0])
				v=int(strlist[1])
				weight=float(strlist[2])
				prob=float(strlist[3])
				G.add_edge(u,v,weight=weight,prob=prob)				
					#G.add_edge(v,u)	
		
		model=KeyedVectors.load(self.embedding)
		self.edges_embs={}
		if (self.embedder=="Hadamard"):	
			for u,v in G.edges():
				if (u<=v):
					self.edges_embs[(u,v)]=model[str(u)]*model[str(v)]*G[u][v]['prob']
				else:
					self.edges_embs[(v,u)]=model[str(u)]*model[str(v)]*G[u][v]['prob']

		if (self.embedder=="Mean"):	
			for u,v in G.edges():
				if (u<=v):
					self.edges_embs[(u,v)]=(model[str(u)]+model[str(v)])/2*G[u][v]['prob']
				else:
					self.edges_embs[(v,u)]=(model[str(u)]+model[str(v)])/2*G[u][v]['prob']	

		if (self.embedder=="L1"):	
			for u,v in G.edges():
				if (u<=v):
					self.edges_embs[(u,v)]=abs(model[str(u)]-model[str(v)])*G[u][v]['prob']
				else:
					self.edges_embs[(v,u)]=abs(model[str(u)]-model[str(v)])*G[u][v]['prob']							

		if (self.embedder=="L2"):	
			for u,v in G.edges():
				if (u<=v):
					self.edges_embs[(u,v)]=abs(model[str(u)]-model[str(v)])**2*G[u][v]['prob']
				else:
					self.edges_embs[(v,u)]=abs(model[str(u)]-model[str(v)])**2*G[u][v]['prob']	

		return G

	def length(self,candidate):
		length=0
		for i in range(1,len(candidate)):
			length+=self.G[candidate[i-1]][candidate[i]]['weight']
		return length

	def checkExistence(self,pathList,P,S,indexList,pathProbs):
		index=random.choices(indexList, weights = pathProbs,k=1 )
		index=index[0]
		G=set()
		L=set()
		for i in range(1,len(pathList[index])):
			if ((  (pathList[index][i-1],pathList[index][i]) not in P ) and (  (pathList[index][i],pathList[index][i-1]) not in P )):
				G.add( (pathList[index][i-1],pathList[index][i]))
				L.add( (pathList[index][i-1],pathList[index][i]))

		for i in range(index):
			terminate=True
			for j in range(1,len(pathList[i])):
				u,v=pathList[i][j-1],pathList[i][j]
				if ((not (u,v) in P)  and (not (v,u) in P)  ):
					if ((not (u,v) in L) and (not (v,u) in L)):
						L.add((u,v))
						if (random.random()<=self.G[u][v]['prob']):
							G.add((u,v))

					if ((not (u,v) in G) and (not (v,u) in G)  ):
		
						terminate=False
						break

			if (terminate==True):
				return 0

		return 1


	def estimateSPProb(self,pathList,targetPath):
		count=0
		S=0
		P=set()
		XP=1
		indexList=[i for i in range(len(pathList))]
		for i in range(1,len(targetPath)):
			P.add(( targetPath[i-1],targetPath[i]))
			XP=XP*self.G[targetPath[i-1]][targetPath[i]]['prob']
		
		pathProbs=[]
		for path in pathList:
			pathprob=1
			for i in range(1,len(path)):
					if (( (path[i-1],path[i]) not in P ) and ( (path[i],path[i-1]) not in P ) ):
							pathprob=pathprob*self.G[path[i-1]][path[i]]['prob']

			pathProbs.append(pathprob)
			S+=pathprob

	
		for _ in range(self.N):
			count+=self.checkExistence(pathList,P,S,indexList,pathProbs)

		
		return (1-float(count)/self.N*S)*XP


	def _predictC(self):
		# no need for dropout when we have batchnorm
		prediction = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.embholder, self.mlp1), training=False ))
		#prediction = tf.add(tf.matmul(prediction,self.w2),self.bias2)
		prediction = tf.matmul(prediction,self.mlp2)
		self.prediction=tf.reshape(prediction,[-1])

	def walkEmbedder(self,path):
		edge=(path[0],path[1]) if path[0]<=path[1] else (path[1],path[0])
		walkEmb=np.copy(self.edges_embs[edge])
		for j in range(2,len(path)):
			edge=(path[j-1],path[j]) if path[j-1]<=path[j] else (path[j],path[j-1])
			walkEmb+=self.edges_embs[edge]
		return walkEmb


		
	def precomputeEmbedding(self,candidates):
		candEmb=[]
		for i in range(len(candidates)):
			candEmb.append(self.walkEmbedder(candidates[i]))
		return candEmb	

	

	def learningSPProb(self,candidates,targetIndex,P,XP,accEmb,p_hat):

		for i in range(1,targetIndex):
			walkEmb=np.copy(self.candEmb[i])
			path=candidates[i]
			pathprob=1
			for j in range(1,len(path)):
					if ( (path[j-1],path[j]) not in P  ):
						pathprob=pathprob*self.G[path[j-1]][path[j]]['prob']
						
					else:
						edge=(path[j-1],path[j]) if path[j-1]<=path[j] else (path[j],path[j-1])
						walkEmb-=self.edges_embs[edge]

			
			embeds=[]
			embeds.append(np.concatenate((accEmb/i ,walkEmb)))
			out=self.sess.run([self.prediction],feed_dict={self.embholder:embeds})
			p_hat+=pathprob*out[0]
			accEmb+=walkEmb

		return (1-p_hat)*XP


	def getAccEmd(self,path,P):
		accEmb=np.copy(self.candEmb[0])

		for i in range(1,len(path)):
			if ( (path[i-1],path[i]) not in P ):
				edge=(path[i-1],path[i]) if path[i-1]<=path[i] else (path[i],path[i-1])
				accEmb-=self.edges_embs[edge]

		return accEmb

	def getProbs(self,candidates,choice='traditional'):
		start=time.time()
		candidates.sort(key=lambda x: self.length(x))
		SPProbs=[]
		first_prob=1
		for i in range(1,len(candidates[0])):
			first_prob=first_prob*self.G[candidates[0][i-1]][candidates[0][i]]['prob']		

		if (choice=="traditional"):
			self.probDict={}
			self.probDict[0]=first_prob


		if (choice=="learning"):
			self.candEmb=self.precomputeEmbedding(candidates)

		SPProbs.append((0,first_prob))  

		prob=0
		for i in range(1,len(candidates)):
			if (choice=="traditional"):
				prob=self.estimateSPProb(candidates[0:i],candidates[i])
			elif (choice=="learning"):
				P=set()
				XP=1
				targetPath=candidates[i]
				for j in range(1,len(targetPath)):
					P.add(( targetPath[j-1],targetPath[j]))
					P.add(( targetPath[j],targetPath[j-1]))
					XP=XP*self.G[targetPath[j-1]][targetPath[j]]['prob']				

				accEmb=self.getAccEmd(candidates[0],P)
				prob=self.learningSPProb(candidates,i,P,XP,accEmb,first_prob)

				#print (prob)
			if (choice=="traditional"):
				self.probDict[i]=prob
			SPProbs.append( (i, prob))

		phase2=time.time()-start

		SPProbs.sort(key=lambda x: x[1], reverse=True)

		if (choice=="traditional"):
			self.rankDict={}
			rank=1
			for index, prob in SPProbs:
				self.rankDict[index]=rank
				rank+=1

		prob=0
		indexsum=0
		#print (choice, SPProbs)
		

		kset=set(self.kvalues)
		probresults=[]
		indexsumresults=[]

		for i in range(self.kvalues[-1]):
			if (i<len(SPProbs)):
				index=SPProbs[i][0]
				prob+=self.probDict[index]
				indexsum+=self.rankDict[index]
				if ((i+1) in kset):
					probresults.append(prob/(i+1))
					indexsumresults.append(indexsum)
			else:
				if ((i+1) in kset):
					probresults.append(0)
					indexsumresults.append(0)
					if (choice=="traditional"):
						self.kcounters[i+1]-=1
			#print (choice,self.probDict[index],SPProbs[i][-1])


		worseprobresults=[]
		if (choice=="traditional"):
			for kvalue in self.kvalues:
				re=0
				if (kvalue<=len(SPProbs)):
					for j in range(len(SPProbs)-1,len(SPProbs)-kvalue-1,-1):
						re+=SPProbs[j][1]
					worseprobresults.append(re/kvalue)
				else:
					worseprobresults.append(0)

		if (choice=="traditional"):
			return probresults,indexsumresults,worseprobresults,phase2

		return probresults,indexsumresults,phase2

	def randomGuess(self, candidates):
		indexlist=[i for i in range(len(candidates))]
		randomloop=5
		probresults=[0 for _ in range(len(self.kvalues))]
		indexsumresults=[0 for _ in range(len(self.kvalues))]
		start=time.time()

		for i in range(len(self.kvalues)):
			if (self.kvalues[i]>len(indexlist)):
				probresults[i]=0
				indexsumresults[i]=0
				continue

			for _ in range(randomloop):
				sampled=random.sample(indexlist,self.kvalues[i])
				indexsum=0
				prob=0
				for idx in sampled:
					prob+=self.probDict[idx]
					indexsum+=self.rankDict[idx]
				probresults[i]+=prob
				indexsumresults[i]+=indexsum

			probresults[i]=probresults[i]/(randomloop*self.kvalues[i])
			indexsumresults[i]=indexsumresults[i]/(randomloop)

		return probresults,indexsumresults,time.time()-start
	def update(self,store,results):

		for i in range(len(results)):
			if (results[i]>0):
				store[i].append(results[i])

	def average(self,store):
		for i in range(len(self.kvalues)):
			value=self.kvalues[i]
			store[i]=store[i]/self.kcounters[value]

	def getQuartilesandAvg(self, store):
		quartiles_avg=[]


		for result in store:
			stats=numpy.quantile(result, [0,0.25,0.5,0.75,0.8,0.9,0.95,1])
			stats=stats.tolist()
			stats.append(sum(result)/len(result))
			
			if (stats):
				stats=[round(stat,10) for stat in stats]
				quartiles_avg.append(stats)
			else:
				quartiles_avg.append([])


		return quartiles_avg
	def postprocess(self,result):

		for i in range(len(result)):
			result[i]=result[i][-1]

		return result

	def computeProbs(self):


		tra_time=learn1_time=learn2_time=random_time=0

		prob_worse=[[] for _  in range(len(self.kvalues))]
		prob_tra=[[] for _  in range(len(self.kvalues))]
		prob_random=[[] for _  in range(len(self.kvalues))]
		prob_learn1=[[] for _  in range(len(self.kvalues))]

		index_tra=[[] for _  in range(len(self.kvalues))]
		index_random=[[] for _  in range(len(self.kvalues))]
		index_learn1=[[] for _  in range(len(self.kvalues))]

		counter=0
		with open(self.paths) as f:	
			candidates=[]
			phase1=0
			for line in f:
				if (counter==self.counter):
					break
				strlist = line.split()
				if (strlist and strlist[0]=='#'):
					if (candidates):
						#print (candidates)
						counter+=1
						probresults_tra,indexresults_tra,worseresults,phase2_tra=self.getProbs(candidates,choice='traditional')
						probresults_learn1,indexresults_learn1,phase2_learn1=self.getProbs(candidates,choice='learning')
						probresults_random,indexresults_random,phase2_random=self.randomGuess(candidates)
						
						self.update(prob_worse,worseresults)
						self.update(prob_tra,probresults_tra)
						self.update(prob_learn1,probresults_learn1)
						self.update(prob_random,probresults_random)

						self.update(index_tra,indexresults_tra)
						self.update(index_learn1,indexresults_learn1)
						self.update(index_random,indexresults_random)

						tra_time+=phase2_tra
						learn1_time+=phase2_learn1
						random_time+=phase2_random

						#print ("traditional:",prob_tra,phase2_tra)
						#print ("learning1:",prob_learn1,phase2_learn1)
						#print ("learning2:",prob_learn2,phase2_learn2)
					candidates=[]
					continue
				if (len(strlist)==1):
					phase1+=float(strlist[0])
					continue
				if (len(strlist)>1):
					candidate=[]
					if (len(candidates)==self.size):
						continue

					for node in strlist:
						candidate.append(int(node))
					candidates.append(candidate)

		prob_worse=self.getQuartilesandAvg(prob_worse)
		prob_tra=self.getQuartilesandAvg(prob_tra)
		prob_learn1=self.getQuartilesandAvg(prob_learn1)
		prob_random=self.getQuartilesandAvg(prob_random)
		self.postprocess(prob_worse)
		self.postprocess(prob_tra)
		self.postprocess(prob_learn1)
		self.postprocess(prob_random)
		index_tra=self.getQuartilesandAvg(index_tra)
		index_learn1=self.getQuartilesandAvg(index_learn1)
		index_random=self.getQuartilesandAvg(index_random)

		original_stdout = sys.stdout  	
		model="model:"+self.trainedGraph+self.suffix
		target="target:"+self.targetGraph
		print (target,model)
		print ("phase 1 time:",phase1/counter," k-values:",self.kvalues)
		print ("RR - prob_avg:", prob_worse)
		print ("FPA - prob_avg:", prob_tra," ranksum_percentile_avg:", index_tra," time:",str(tra_time/counter)  )
		print ("Random - prob_avg:", prob_random," ranksum_percentile_avg:", index_random," time:",str(random_time/counter)   )
		print ("LPA - prob_avg:", prob_learn1," ranksum_percentile_avg:",index_learn1," time:",str(learn1_time/counter)   )		
if __name__ == "__main__":
	args = parse_args()
	mpsp=MPSP(args)
	mpsp.computeProbs()
