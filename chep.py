import numpy as np
import	os
from optparse import OptionParser
import	sys
from sklearn.cluster import KMeans
#from decimal import Decimal
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from scipy.spatial.distance import cdist

progname = os.path.basename(sys.argv[0])
parser = OptionParser(usage="usage: %prog --infile relion.star --header=40 --class_col=29 --helix_col=3 --cluster_num=4 --elbow_clusters=11")
parser.add_option("--infile", type="string", default="./", help="Pass the 2D classification Relion star file")
parser.add_option("--helix_col", type="int", default="3", help="Column number of _rlnHelicalTubeID in the star file")
parser.add_option("--class_col", type="int", default="29", help="Column number of _rlnClassNumber in the star file")
parser.add_option("--header", type="int", default="40", help="Number of header lines to be removed from the star file")
parser.add_option("--cluster_num", type="int", default="2", help="Number of clusters for k-means algorithm")
parser.add_option("--elbow_clusters", type="int", default="11", help="Maximum number of clusters for Elbows test")
(options,args) = parser.parse_args()

#reading data from the given file
data=np.loadtxt(options.infile, dtype='int', skiprows=options.header, usecols=(options.helix_col-1,options.class_col-1))

# find the number of particles in the given dataset
ptcl_nr=(np.size(data, 0))
helix_list= np.zeros((ptcl_nr,1))
print("Number of particles in the dataset: " + str(ptcl_nr))

#renaming the helical protein IDs
helix_id=1
ptcl_id=data[0,0]

for ptcl in range(ptcl_nr):
	if ptcl_id==data[ptcl,0]:
		helix_list[ptcl]=helix_id
	else: 
		ptcl_id!=data[ptcl,0]  
		helix_id=helix_id+1
		helix_list[ptcl]=helix_id
		ptcl_id=data[ptcl,0]

data=np.hstack((data,helix_list))

# find the range of 2D class numbers and helical proteins
class_max, helix_max = data[:,[1,2]].max(axis=0)
class_min, helix_min = data[:,[1,2]].min(axis=0)
print("The number of 2D classes in the given dataset is: " + str(int(class_min)) + " - " + str(int(class_max)))
print("The number of Helical proteins in the given dataset is: " + str(int(helix_min)) + " - " + str(int(helix_max)))


#create a matrix with zeros containing dimensions of helix_max, class_max
fc_matrix=np.zeros((int(helix_max), int(class_max)))
fc_matrix_norm=np.zeros((int(helix_max), int(class_max)))


# convert the data into matrix form with each helical protein as a vector
for i in range(int(helix_max)):
	for j in range(int(class_max)):
		helix_vec=(data[(data[:,2] == i+helix_min) & (data[:,1] == j+class_min)])
		fc_matrix[i,j]=(np.float(np.size(helix_vec[:,1])))
	fc_matrix_norm[i,:] = fc_matrix[i,:] / np.sum(fc_matrix[i,:])

# k-means clustering for the fc_matrix_norm
kmeans = KMeans(n_clusters=options.cluster_num, random_state=20).fit(fc_matrix_norm)
y_kmeans = kmeans.predict(fc_matrix_norm)
print("kmeans clustering")

#default plotted on first two principal compoenents
sklearn_pca = sklearnPCA(n_components=2)
y_pca = sklearn_pca.fit_transform(fc_matrix_norm)

#validating the number of clusters with Elbow test
SSE = []
K = range(1,options.elbow_clusters)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(fc_matrix_norm)
    kmeanModel.fit(fc_matrix_norm)
    SSE.append(sum(np.min(cdist(fc_matrix_norm, kmeanModel.cluster_centers_, 'euclidean'), axis=1)))

#writing output file
 
np.savetxt('cluster.log', y_kmeans, fmt=['%d'], header='#Cluster assigment for the helical proteins')

print("#### Completed #######")

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
