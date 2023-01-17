# tensorflow:
# problem solved: reproducibility of node2vec, peprocess random walk, batch norm and dropout, objective function, presave node2vec (troublesome with incomplte api), wheter use weighted information in node2vec.
# how to set the edge prob. cannot just set the same.
# save/load the model.
import sys
import random
import networkx as nx
import argparse
import heapq
import time
import os
import numpy as np
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
import tensorflow as tf
import logging
from gensim.models import KeyedVectors
import shutil

#from numpy.random import choice


def parse_args():
	parser = argparse.ArgumentParser(description="learning procedure")

	parser.add_argument('--graph', nargs='?', default='BJ', help='input graph file')

	parser.add_argument('--N', type=int, default='100',help='the number of monte carlo simulation')

	parser.add_argument('--m', type=int, default='20',help='the number of random walk')

	parser.add_argument('--ratio', type=float, default='0.1',help='the percentage of graph size for sampling starting nodes')

	parser.add_argument('--length', type=int, default='30',help='the length of walk')

	parser.add_argument('--embedder', nargs='?', default='L1', help='Hadamard, Mean, L1 and L2')
	
	parser.add_argument('--dimensions', type=int, default='128',help='the number of embedding dimensions')

	parser.add_argument('--batchsize', type=int, default='256',help='the batch size in each training iteration')	

	parser.add_argument('--counter', type=int, default='100',help='the number of validation pairs')

	return parser.parse_args()

class MPSP(object):
	"""docstring for MPSP"""
	

	def __init__(self, args):
		super(MPSP, self).__init__()
		self.suffix="_op"+str(args.embedder)+"_m"+str(args.m)+"_l"+str(args.length)+"_r"+str(args.ratio)
		self.m=args.m
		self.length=args.length
		self.input=args.graph
		self.N=args.N	
		self.walks=[]
		self.embedder=args.embedder
		self.dimensions=args.dimensions
		self.batchsize=args.batchsize
		self.counter=args.counter
		self.validation_set="validation/"+self.input

		self.optimizer=tf.train.AdamOptimizer()
		self.embholder=tf.placeholder(dtype=tf.float32,shape=(None,None))
		self.probholder=tf.placeholder(dtype=tf.float32,shape=(None))
		self.mlp1 = self.glorot([self.dimensions*2, self.dimensions], name='mlp1')
		self.mlp2 = self.glorot([self.dimensions,1], name='mlp2')	
		self.loss=0
		self.op=None
		self.loadGraph()
		self.targets=int(args.ratio*len(self.G))
		self._predictC()
		self._loss()

	def loadGraph(self):
		self.G=nx.Graph()

		with open(self.input) as f:	
			for line in f:
				strlist = line.split()

				u=int(strlist[0])
				v=int(strlist[1])
				weight=float(strlist[2])
				prob=float(strlist[3])
				self.G.add_edge(u,v,weight=weight,prob=prob)		

					#G.add_edge(v,u)				

		embeddingfile="embedding/"+self.input
		model=KeyedVectors.load(embeddingfile)

		self.edges_embs={}
		if (self.embedder=="Hadamard"):	
			for u,v in self.G.edges():
				if (u<=v):
					self.edges_embs[(u,v)]=model[str(u)]*model[str(v)]*self.G[u][v]['prob']
				else:
					self.edges_embs[(v,u)]=model[str(u)]*model[str(v)]*self.G[u][v]['prob']		


		if (self.embedder=="Mean"):	
			for u,v in self.G.edges():
				if (u<=v):
					self.edges_embs[(u,v)]=(model[str(u)]+model[str(v)])/2*self.G[u][v]['prob']
				else:
					self.edges_embs[(v,u)]=(model[str(u)]+model[str(v)])/2*self.G[u][v]['prob']	

		if (self.embedder=="L1"):	
			for u,v in self.G.edges():
				if (u<=v):
					self.edges_embs[(u,v)]=abs(model[str(u)]-model[str(v)])*self.G[u][v]['prob']
				else:
					self.edges_embs[(v,u)]=abs(model[str(u)]-model[str(v)])*self.G[u][v]['prob']							

		if (self.embedder=="L2"):	
			for u,v in self.G.edges():
				if (u<=v):
					self.edges_embs[(u,v)]=abs(model[str(u)]-model[str(v)])**2*self.G[u][v]['prob']
				else:
					self.edges_embs[(v,u)]=abs(model[str(u)]-model[str(v)])**2*self.G[u][v]['prob']	


	def randomWalk(self,target):
		self.walks=[]
		#print ("neighors",self.G.neighbors(target))
		for _ in range(self.m):
			walk=[target]
			for _ in range(self.length):
				neighbor=random.choice(self.G.neighbors(target))
				walk.append(neighbor)
				target=neighbor
			self.walks.append(walk)
		#print (self.walks)

	def groundtruth(self,index):
		targetWalk=self.walks[index]

		C=0

		for _ in range(self.N):
			G=set()
			L=set()
			for i in range(1,len(targetWalk)):
				edge=(targetWalk[i-1],targetWalk[i])
				G.add(edge)	
				L.add(edge)
			add=True


			for i in range(len(self.walks)):
				if (i==index):
					continue

				terminate=True
				for j in range(1,len(self.walks[i])):
					u,v=self.walks[i][j-1],self.walks[i][j]
					if ( (not (u,v) in L) and (not (v,u) in L) ):
						L.add((u,v))
						if (random.random()<=self.G[u][v]['prob']):
							G.add((u,v))
					if ((not (u,v) in G) and (not (v,u) in G)):
						terminate=False
						break

				if (terminate==True):
					add=False
					break

			if (add):
				C+=1


		return float(C)/self.N


	def walkEmbedder(self,index):
		path=self.walks[index]
		edge=(path[0],path[1]) if path[0]<=path[1] else (path[1],path[0])
		walkEmb=np.copy(self.edges_embs[edge])
		for j in range(2,len(path)):
			edge=(path[j-1],path[j]) if path[j-1]<=path[j] else (path[j],path[j-1])
			walkEmb+=self.edges_embs[edge]
		return walkEmb

	def summation(self):
		
		sumEmd=np.zeros(self.dimensions)
		for index in range(len(self.walks)):
			sumEmd+=self.walkEmbedder(index)
		return sumEmd

	def glorot(self,shape, name=None):
		"""Glorot & Bengio (AISTATS 2010) init."""
		init_range = np.sqrt(6.0/(shape[0]+shape[1]))
		initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
		return tf.Variable(initial, name=name)

	def zeros(self,shape, name=None):
		"""All zeros."""
		initial = tf.zeros(shape, dtype=tf.float32)
		return tf.Variable(initial, name=name)

	def getTrainingPairs(self):
		self.embeds=[]
		self.probs=[]
		counter=0
		while (counter<self.targets):
			counter+=1
			target=random.choice(self.G.nodes())
			self.randomWalk(target)
			sumEmb=self.summation()	
			for index in range(len(self.walks)):
				prob=self.groundtruth(index)
				walkEmb=self.walkEmbedder(index)
				aggEmb=(sumEmb-walkEmb)/(len(self.walks)-1)
				self.embeds.append(np.concatenate((aggEmb,walkEmb)))
				self.probs.append(prob)


	def _loss(self):
		prediction = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.embholder, self.mlp1), training=True ))
		prediction = tf.matmul(prediction,self.mlp2)
		self.prediction=tf.reshape(prediction,[-1])

		self.loss=tf.keras.losses.mean_absolute_error(self.prediction,self.probholder)
		
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

		with tf.control_dependencies(update_ops):
			self.op=self.optimizer.minimize(self.loss)

	
###################################validation-start#####################################

	def compute_length(self,candidate):
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

	
		for _ in range(self.N_val):
			count+=self.checkExistence(pathList,P,S,indexList,pathProbs)

		
		return (1-float(count)/self.N_val*S)*XP


	def _predictC(self):
		# no need for dropout when we have batchnorm
		prediction = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.embholder, self.mlp1), training=False ))
		#prediction = tf.add(tf.matmul(prediction,self.w2),self.bias2)
		prediction = tf.matmul(prediction,self.mlp2)
		self.val_prediction=tf.reshape(prediction,[-1])

	def walkEmbedder2(self,path):
		edge=(path[0],path[1]) if path[0]<=path[1] else (path[1],path[0])
		walkEmb=np.copy(self.edges_embs[edge])
		for j in range(2,len(path)):
			edge=(path[j-1],path[j]) if path[j-1]<=path[j] else (path[j],path[j-1])
			walkEmb+=self.edges_embs[edge]
		return walkEmb


		
	def precomputeEmbedding(self,candidates):
		candEmb=[]
		for i in range(len(candidates)):
			candEmb.append(self.walkEmbedder2(candidates[i]))
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
			out=self.sess.run([self.val_prediction],feed_dict={self.embholder:embeds})
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
		candidates.sort(key=lambda x: self.compute_length(x))
		SPProbs=[]
		first_prob=1
		for i in range(1,len(candidates[0])):
			first_prob=first_prob*self.G[candidates[0][i-1]][candidates[0][i]]['prob']		

		if (choice=="traditional"):
			self.probDict={}
			self.probDict[0]=first_prob


		if (choice=="learning1"):
			self.candEmb=self.precomputeEmbedding(candidates)

		SPProbs.append((0,first_prob))  

		prob=0
		for i in range(1,len(candidates)):
			if (choice=="traditional"):
				prob=self.estimateSPProb(candidates[0:i],candidates[i])
			elif (choice=="learning1"):
				P=set()
				XP=1
				targetPath=candidates[i]
				for j in range(1,len(targetPath)):
					P.add(( targetPath[j-1],targetPath[j]))
					P.add(( targetPath[j],targetPath[j-1]))
					XP=XP*self.G[targetPath[j-1]][targetPath[j]]['prob']				

				accEmb=self.getAccEmd(candidates[0],P)
				prob=self.learningSPProb(candidates,i,P,XP,accEmb,first_prob)

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


		return probresults,indexsumresults,phase2


	def update(self,store,results):

		for i in range(len(results)):
			store[i]+=results[i]

	def average(self,store):
		for i in range(len(self.kvalues)):
			value=self.kvalues[i]
			store[i]=store[i]/self.kcounters[value]


	def validation(self):
		self.candidate_size=20
		self.kvalues=[1,5,10]
		
		tra_time=learn1_time=0
		self.N_val=1000
		self.kcounters={}
		for value in self.kvalues:
			self.kcounters[value]=self.counter

		prob_tra=[0 for _  in range(len(self.kvalues))]
		prob_learn1=[0 for _  in range(len(self.kvalues))]


		index_tra=[0 for _  in range(len(self.kvalues))]
		index_learn1=[0 for _  in range(len(self.kvalues))]

		counter=0
		with open(self.validation_set) as f:	
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
						probresults_tra,indexresults_tra,phase2_tra=self.getProbs(candidates,choice='traditional')
						probresults_learn1,indexresults_learn1,phase2_learn1=self.getProbs(candidates,choice='learning1')
						
						self.update(prob_tra,probresults_tra)
						self.update(prob_learn1,probresults_learn1)

						self.update(index_tra,indexresults_tra)
						self.update(index_learn1,indexresults_learn1)

						tra_time+=phase2_tra
						learn1_time+=phase2_learn1

					candidates=[]
					continue
				if (len(strlist)==1):
					phase1+=float(strlist[0])
					continue
				if (len(strlist)>1):
					candidate=[]
					if (len(candidates)==self.candidate_size):
						continue

					for node in strlist:
						candidate.append(int(node))
					candidates.append(candidate)

		self.average(prob_tra)
		self.average(prob_learn1)
		self.average(index_tra)
		self.average(index_learn1)

		return prob_tra,index_learn1,prob_learn1
###############################################validation-end######################################################
	
	def compute(self):
		start=time.time()

		var_list = [var for var in tf.global_variables() if "moving" in var.name]
		var_list += tf.trainable_variables()
		saver = tf.train.Saver(var_list=var_list,save_relative_paths=True,max_to_keep=50)


		outstring=""
		converged=False
		checkpoint=0
		threshold=3
		epochs=0
		preloss=0
		bestepoch=1
		maxprob=0

		with tf.Session() as self.sess:
			self.sess.run(tf.global_variables_initializer())
			self.sess.run(tf.local_variables_initializer())
			s1=time.time()
			self.getTrainingPairs()
			walktime=time.time()-s1

			trainingIndex=[x for x in range(len(self.embeds))]

			while (not converged):
				epochs+=1
				counter=0
				embeds,probs=[],[]
				loss=0
				step=0
				while(counter<len(self.embeds)):
					counter+=1
					embeds.append(self.embeds[trainingIndex[counter-1]])
					probs.append(self.probs[trainingIndex[counter-1]])

					if (counter%self.batchsize==0 or counter==len(self.embeds)):
						#probs=[probs]
						step+=1
						out=self.sess.run([self.loss,self.op],feed_dict={self.embholder:embeds,self.probholder:probs})
						embeds,probs=[],[]
						loss+=out[0]

				if (preloss>0 and abs(loss-preloss)/preloss<0.1):
					checkpoint+=1

				if (preloss>0 and abs(loss-preloss)/preloss>=0.1):
					checkpoint=0

				if (checkpoint==threshold):
					converged=True

				preloss=loss

				groundtruth,validation_index,validation_prob=self.validation()
				
				if (validation_prob[0]>maxprob):
					maxprob=validation_prob[0]
					bestepoch=epochs

				print (self.input+self.suffix, "epochs:",epochs," loss:",loss/step," groundtruth:",groundtruth," validation_index:",validation_index," validation_prob:",validation_prob)

				random.shuffle(trainingIndex)


				directory="models/"+self.input+self.suffix+"_epoch"+str(epochs)+"/"

				if not os.path.exists(directory):
			  
			  		# Create a new directory because it does not exist 
			  		os.makedirs(directory)

				save_path = saver.save(self.sess,directory+"model")

			for i in range(1,epochs+1):
				directory="models/"+self.input+self.suffix+"_epoch"+str(i)+"/"
				if os.path.exists(directory) and os.path.isdir(directory):
					if (i!=bestepoch):
						shutil.rmtree(directory)
					else:
						#print ("saving best epoch!", directory)
						newdirectory="models/"+self.input+self.suffix+"/"
						if os.path.exists(newdirectory) and os.path.isdir(newdirectory):
							shutil.rmtree(newdirectory)
						os.rename(directory,newdirectory)


			print (self.input+self.suffix,"total_runtime:",time.time()-start," training_set_time::",walktime," prediction_time", time.time()-start-walktime," total_epochs:",epochs," best_epoch",bestepoch)
			print (" ")


if __name__ == "__main__":
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	args = parse_args()
	mpsp=MPSP(args)
	mpsp.compute()
