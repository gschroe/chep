#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:09:22 2019

@author: K.pothula, J.Geraets, G.Schroeder
"""

import numpy as np
import	os
from optparse import OptionParser
import	sys
from sklearn.cluster import KMeans
from decimal import Decimal
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from scipy.spatial.distance import cdist

progname = os.path.basename(sys.argv[0])
parser = OptionParser(usage="usage: %prog --infile relion.star --header=40 --class_col=29 --helix_col=3 --cluster_num=4 --elbow_clusters=11")
parser.add_option("--infile", type="string", default="./", help="Pass the 2D classification Relion star file")
parser.add_option("--cluster_num", type="int", default="3", help="Number of clusters for k-means algorithm")
parser.add_option("--elbow_clusters", type="int", default="11", help="Maximum number of clusters for Elbows test")
(options,args) = parser.parse_args()

class read_relion(object):

    def __init__(self, file):
        self.file = file
        
        
    def getRdata(self):
        Rvar = [] #read the variables 
        Rdata = [] # read the data

        for star_line in open(self.file).readlines():
            if star_line.find("_rln") != -1:
                var = star_line.split()
                Rvar.append(var[0])
            #    Rvar_len = Rvar_len+1
            elif star_line.find("data_") != -1 or star_line.find("loop_") != -1 or len(star_line.strip()) == 0:
                continue
                
            else:
                Rdata.append(star_line.split())
                
        return Rvar, Rdata       

class write_relion(object):
    def __init__(self, file_nr, cluster_info):
        self.file_nr=file_nr
        self.cluster_info=cluster_info
        
    def writemetadata(self):
            filename = 'cluster' + str(self.file_nr) + '.star'
            with open(filename, "w") as file:
                file.writelines("%s\n" % "          ")
                file.writelines("%s\n" % "data_")
                file.writelines("%s\n" % "           ")
                file.writelines("%s\n" %  "loop_")

                for item in self.cluster_info:
                   # fullstr = ' '.join([str(elem) for elem in item ])
                    file.writelines("%s\n" % item)
    
    def writecluster(self):
            filename = 'cluster' + str(self.file_nr) + '.star'
            with open(filename, "a") as file:
                for item in self.cluster_info:
                    fullstr = '  '.join([str(elem) for elem in item ])
                    file.writelines("%s\n" % fullstr)      

relion_data = read_relion(options.infile)

if len(relion_data.getRdata()[0])==(len(relion_data.getRdata()[1][0])):
    rlndata=np.asarray(relion_data.getRdata()[1])
else:
    print("###################################################################################")
    print("          ")
    print("          ")
    print("The number of variables doesn't match with the number of columns in the star file" )
    print("          ")
    print("          ")
    print("The job gets killed")
    print("          ")
    print("          ")
    print("###################################################################################")
    
Rvar_nr=len(relion_data.getRdata()[0])
print("Number of parameters in the relion file = " + str(int(Rvar_nr)))

ptcl_nr=len(relion_data.getRdata()[1]) # no of particles in the relion file 
print("Number of particles in the relion file = " + str(int(ptcl_nr)))


rlntubeID_col = relion_data.getRdata()[0].index( '_rlnHelicalTubeID' )
rlnclassID_col = relion_data.getRdata()[0].index( '_rlnClassNumber' )
rlnMicrographName_col = relion_data.getRdata()[0].index( '_rlnMicrographName' )

helixID_col = len(relion_data.getRdata()[1])

#initialising array
helixIDclass_data= np.zeros((ptcl_nr,2))

helixdict = {}
counter = 1

# renaming the helical_id 
for ptcl in range(ptcl_nr):
    uniqueID = rlndata[ptcl,rlnMicrographName_col] + '-' + rlndata[ptcl,rlntubeID_col]
    #check whether ID exists in dictionary
    if uniqueID in helixdict:
        helixID = helixdict[uniqueID]
    else:
        #add unseen ID to the dictionary
        helixID = counter
        counter += 1
        helixdict[uniqueID] = helixID
    helixIDclass_data[ptcl,0]=np.int_(helixID)

 

print("Number of Full-length helical proteins = " + str(int(helixID)))



helix_list= np.zeros((ptcl_nr,1))
print("Number of particles in the dataset: " + str(ptcl_nr))


rlndata=np.append(rlndata,helixIDclass_data, axis=1)
helixIDclass_data[:,1]=np.int_(rlndata[:,rlnclassID_col])
rlnclassID_max = (np.int_(rlndata[:,[rlnclassID_col]]).max(axis=0))
rlnclassID_min = (np.int_(rlndata[:,[rlnclassID_col]]).min(axis=0))
helixID_max = (np.int_(helixIDclass_data[:,0].max(axis=0)))
helixID_min = (np.int_(helixIDclass_data[:,0].min(axis=0)))

#print("Range of 2D classes = " + str(int(rlnclassID_min)) + " - " + str(int(rlnclassID_max)))
print("Number of 2D classes = " + str(len(np.unique(helixIDclass_data[:,1]))))

fc_matrix=np.zeros((int(helixID_max), int(rlnclassID_max)))

for ptcl in range(ptcl_nr):
    f, c = helixIDclass_data[ptcl,0], helixIDclass_data[ptcl,1]
    fc_matrix[int(f-1), int(c-1)] +=1
fc_matrix = fc_matrix / fc_matrix.sum(axis=1,keepdims=True)

kmeans = KMeans(n_clusters=options.cluster_num, random_state=20).fit(fc_matrix)
y_kmeans = kmeans.predict(fc_matrix)
print("kmeans clustering")

#default plotted on first two principal compoenents
sklearn_pca = sklearnPCA(n_components=2)
y_pca = sklearn_pca.fit_transform(fc_matrix)

# writing clusters to files
# an additional comment

for cluster in range(options.cluster_num):
    cluster_data=[]
    count=0
    for ptcl in range(ptcl_nr):
        helixID=int(float((rlndata[ptcl,Rvar_nr])))
        if (y_kmeans[helixID-1])==cluster:
            cluster_data.append(rlndata[ptcl, 0:Rvar_nr])
        else:
            continue

    cluster_meta=write_relion(cluster,relion_data.getRdata()[0])
    cluster_meta.writemetadata()
    cluster_out = write_relion(cluster,cluster_data)
    cluster_out.writecluster()


#validating the number of clusters with Elbow test
SSE = []
K = range(1,options.elbow_clusters)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(fc_matrix)
    kmeanModel.fit(fc_matrix)
    SSE.append(sum(np.min(cdist(fc_matrix, kmeanModel.cluster_centers_, 'euclidean'), axis=1)))


#plotting figures
fig=plt.figure()
ax = fig.add_subplot(121)
plt.scatter(y_pca[:, 0], y_pca[:, 1], c=y_kmeans, alpha=0.2, s=170)
plt.xlabel('PC1', fontsize=18)
plt.ylabel('PC2', fontsize=18)

plt.subplot(122)
plt.plot(K, SSE, '-ok', c='black', alpha=0.5)
plt.xlabel('Cluster ID', fontsize=18)
plt.ylabel('SSE', fontsize=18)
plt.show()
fig.savefig("Cluster.png")
