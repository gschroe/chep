#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# *******************************************************************
# *
# * Authors:  Karunakar Reddy Pothula (k.pothula@fz-juelich.de)
# *           James Alexander Geraets (j.geraets@fz-juelich.de)
# *           Gunnar F. Schroeder
# *
# * This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as
# * published by the Free Software Foundation; either version 2 of
# * the License, or (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *******************************************************************

import numpy as np
import os
import sys
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from decimal import Decimal
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from scipy.spatial.distance import cdist
from itertools import combinations

CHEP_VERSION = "0.1.9"

progname = os.path.basename(sys.argv[0])
parser = ArgumentParser(description="CHEP: Clustering of HElical " +
        "Polymers. Tool to cluster homogeneous helical particles " +
        "from the results of 2D classification.")
parser.add_argument("-i", "--infile", type=str, required=True,
        help="Pass the 2D classification Relion star file")
parser.add_argument("-c", "--cluster_num", type=int, default=2,
        help="Number of clusters for k-means algorithm")
parser.add_argument("-e", "--elbow_clusters", type=int,
        help="Try k-means elbow test for this many clusters")
parser.add_argument("-n", "--pca_num", type=int, default=2,
        help="Number of PCA axes to apply for visualization")
parser.add_argument("-x", "--no_display", action="store_true",
        help="No display - run as command line only")
parser.add_argument("-V", "--version", action="version",
        version="CHEP version: " + CHEP_VERSION,
        help="Print CHEP version and exit")
options = parser.parse_args()


class ReadStar(object):
    '''
    Read in relion data STAR files
    '''
    def __init__(self, infile):
        self.infile = infile
        self._var = None
        self._data = None
        self._optics = None

    def var(self):
        '''
        Variables within the particle data loop of the relion STAR
        file
        '''
        if self._var:
            return self._var
        else:
            self._parse_file()
            return self._var

    def data(self):
        '''
        Data loop of the relion STAR file as a list
        '''
        if self._data:
            return self._data
        else:
            self._parse_file()
            return self._data

    def data_array(self, nested=False):
        '''
        Data loop of the relion STAR file as an array
        '''
        if self._data:
            if (len(self._var) == len(self._data[0])):
                return np.asarray(self._data)
            else:
                print("ERROR: Number of metadata variables does" +
                        "not match number of columns in the input" +
                        "STAR file")
                sys.exit(2)
        elif nested:
            print("ERROR: Recursion error when reading file")
            sys.exit(2)
        else:
            self._parse_file()
            return self.data_array(nested=True)

    def optics(self):
        '''
        New relion STAR file format includes optics table, 
        accessible by this method.
        '''
        if self._optics:
            return self._optics
        else:
            self._parse_file()
            return self._optics

    def _parse_file(self):
        var = [] #read the variables
        data = [] # read the data
        optics = [] # store new relion optics data table

        new_relion_star_format = False

        for line in open(self.infile).readlines():
            if line[0] == "#":
                continue
            elif "data_optics" in line: 
                new_relion_star_format = "OPTICS"
            elif "data_particles" in line:
                new_relion_star_format = "PARTICLES"

            if new_relion_star_format == "OPTICS":
                optics.append(line.strip())

            elif "_rln" in line:
                var.append(line.split()[0])
            elif ("data_" in line or
                    "loop_" in line or
                    len(line.strip()) == 0):
                continue
            else:
                data.append(line.split())
        self._var = var
        self._data = data
        self._optics = optics


class WriteStar(object):
    '''
    Write out relion-compatible STAR files
    '''
    def __init__(self, file_name, var, data, optics):
        self.file_name = file_name
        self.var = var
        self.data = data
        self.optics = optics

    def write(self):
        '''
        Write out relion-compatible STAR files
        '''
        with open(self.file_name, "w") as outfile:
            if self.optics:
                outfile.write("\n" + "\n".join(self.optics) + "\n")
                outfile.write("\ndata_particles\n\nloop_\n")
            else:
                outfile.write("\ndata_\n\nloop_\n")
            for index, variable in enumerate(self.var):
                outfile.write(variable + " #" + str(index + 1) + "\n")
            for item in self.data:
                fullstr = "\t".join([str(elem) for elem in item])
                outfile.write(fullstr + "\n")


if __name__ == "__main__":
    instar = ReadStar(options.infile)
    rlnvar = instar.var()
    rlndata_list = instar.data()
    rlndata = instar.data_array()
    rlnoptics = instar.optics()

    rlnvar_nr = len(rlnvar)
    print("Number of parameters in the relion file = " +
            str(int(rlnvar_nr)))

    ptcl_nr=len(rlndata_list)
    print("Number of particles in the relion file = " +
            str(int(ptcl_nr)))

    rlntubeID_col = rlnvar.index('_rlnHelicalTubeID')
    rlnclassID_col = rlnvar.index('_rlnClassNumber')
    rlnMicrographName_col = rlnvar.index('_rlnMicrographName')
    rlnAnglePsi_col = rlnvar.index('_rlnAnglePsi')


    #initialising array
    helixIDclass_data = np.zeros((ptcl_nr, 2))

    helixdict = {}
    counter = 1

    # renaming the helical_id
    for ptcl in range(ptcl_nr):
        uniqueID = (rlndata[ptcl, rlnMicrographName_col] +
                '-' + rlndata[ptcl,rlntubeID_col])
        #check whether ID exists in dictionary
        if uniqueID in helixdict:
            helixID = helixdict[uniqueID]
        else:
            #add unseen ID to the dictionary
            helixID = counter
            counter += 1
            helixdict[uniqueID] = helixID
        helixIDclass_data[ptcl,0] = np.int_(helixID)

    print("Number of Full-length helical proteins = " +
            str(int(helixID)))

    helix_list= np.zeros((ptcl_nr,1))
    print("Number of particles in the dataset: " + str(ptcl_nr))

    rlndata = np.append(rlndata, helixIDclass_data, axis=1)
    helixIDclass_data[:,1] = np.int_(rlndata[:,rlnclassID_col])
    rlnclassID_max = np.int_(rlndata[:,[rlnclassID_col]]).max(axis=0)
    rlnclassID_min = np.int_(rlndata[:,[rlnclassID_col]]).min(axis=0)
    helixID_max = np.int_(helixIDclass_data[:,0].max(axis=0))
    helixID_min = np.int_(helixIDclass_data[:,0].min(axis=0))

    print("Number of 2D classes = " +
            str(len(np.unique(helixIDclass_data[:,1]))))

    # get psi angles
    psi_data = np.float_(rlndata[:,[rlnAnglePsi_col]])
    helix_data = helixIDclass_data[:,0]
    class_data = helixIDclass_data[:,1]
    print("hier")
    #print(psi_data)
    #print(helixIDclass_data)

    # initialize distance matrix
    nr_2dclasses = len(np.unique(helixIDclass_data[:,1]))
    i=j=k=0
    distance_matrix = np.zeros((int(nr_2dclasses), int(nr_2dclasses)))


    # TEST 
    fp = open("distance_matrix.dat","rb")
    np.load(fp,distance_matrix)
    fp.close()
    
    sys.exit()


    
    
    # Calculate average psi difference between class_i and class_j      
    for class_i in range(nr_2dclasses - 1):
        print("class_i = " + str(class_i))
        for class_j in range(class_i + 1, nr_2dclasses, 1):
            
            delta_psi = 0.0      
            counter = 0
            
            # find all filaments that have segments in both 2d classes
            list_filaments_in_class_i = []
            list_filaments_in_class_j = []
            list_filaments_in_both_classes = []
            
            for i in range(ptcl_nr):
                #print(int(helixIDclass_data[i,1]), class_i)
                if int(helixIDclass_data[i,1]) == class_i:
                    list_filaments_in_class_i.append(helixIDclass_data[i,0])
                if int(helixIDclass_data[i,1]) == class_j:
                    list_filaments_in_class_j.append(helixIDclass_data[i,0]) 
                    
            list_filaments_in_class_i = np.unique(list_filaments_in_class_i)
            list_filaments_in_class_j = np.unique(list_filaments_in_class_j)
            #print(list_filaments_in_class_i)
            #print(list_filaments_in_class_j)
            for i in range(len(list_filaments_in_class_i)):
                for j in range(len(list_filaments_in_class_j)):
                    if list_filaments_in_class_i[i] == list_filaments_in_class_j[j]:
                        list_filaments_in_both_classes.append(int(list_filaments_in_class_i[i]))
                    
            
            
            for k in list_filaments_in_both_classes:
                # index of all segments from filament i that are in class_i
                id_segments_class_i = np.where((helix_data == k ) &  (class_data == class_i))
                #print(type(id_segments_class_i))
                id_segments_class_j = np.where((helix_data == k ) &  (class_data == class_j))
                #print("id i = " + str(id_segments_class_i) + str(helix_data[id_segments_class_i]))
                #print("id j = " + str(id_segments_class_j))
                #print("id_segments_class_i = " + str(id_segments_class_i[0]))
    
                for i in id_segments_class_i[0]:
                    for j in id_segments_class_j[0]:
                        #print(type(i),i)
                        #print(j)
                        psi_i = (np.asscalar(psi_data[i]) + 180.0 ) % 360.0 - 180.0
                        psi_j = (np.asscalar(psi_data[j]) + 180.0 ) % 360.0 - 180.0             
                        delta_psi += min(abs(psi_i - psi_j), 360.0 - abs(psi_i - psi_j))
                        counter = counter + 1
                        
                
            if (counter>0):            
                #print("delta_psi i j = " + str(delta_psi / float(counter)) + " " + str(i) + " " + str(j))
                distance_matrix[class_i,class_j] = delta_psi / float(counter)
                distance_matrix[class_j,class_i] = distance_matrix[class_i,class_j]

    

    # K-means clustering with K=2 using distance_matrix
    fp = open("distance_matrix.dat","wb")
    np.save(fp,distance_matrix)
    fp.close()




    fc_matrix=np.zeros((int(helixID_max), int(rlnclassID_max)))

    for ptcl in range(ptcl_nr):
        f, c = helixIDclass_data[ptcl,0], helixIDclass_data[ptcl,1]
        fc_matrix[int(f-1), int(c-1)] +=1
    fc_matrix = fc_matrix / fc_matrix.sum(axis=1,keepdims=True)

    kmeans = KMeans(n_clusters=options.cluster_num,
            random_state=20).fit(fc_matrix)
    y_kmeans = kmeans.predict(fc_matrix)
    print("kmeans clustering")
    stats = []
    for cluster in range(max(y_kmeans) + 1):
        stats.append((float(sum(y_kmeans==cluster))/len(y_kmeans),
                cluster))
    stats.sort(reverse=True)
    cluster_ids = {c: i + 1 for i, (s, c) in enumerate(stats)}
    sortkmeans = np.array([cluster_ids[i] for i in y_kmeans])
    stat_list = [str(cluster_ids[j]) + ":" + "{0:.0f}%".format(100*i)
            for i,j in stats]
    print("\t".join(stat_list))

    #default plotted on first two principal components
    try:
        sklearn_pca = sklearnPCA(n_components=options.pca_num)
        y_pca = sklearn_pca.fit_transform(fc_matrix)
    except ValueError as e:
        print("ERROR: PCA failed with " + str(options.pca_num) +
                " components", e)
        sys.exit(2)

    # writing clusters to files
    for cluster in range(1, options.cluster_num + 1):
        cluster_data=[]
        count=0
        for ptcl in range(ptcl_nr):
            helixID=int(float((rlndata[ptcl,rlnvar_nr])))
            if (sortkmeans[helixID-1])==cluster:
                cluster_data.append(rlndata[ptcl, 0:rlnvar_nr])
            else:
                continue

        cstr = str(cluster).zfill(len(str(options.cluster_num)))
        file_name = (os.path.splitext(options.infile)[0] +
                ".chep_k" + str(options.cluster_num) + "_" +
                cstr + ".star")

        cluster_out = WriteStar(file_name, rlnvar, cluster_data, rlnoptics)
        cluster_out.write()

    if options.elbow_clusters:
        subplots = (len(list(combinations(
                range(options.pca_num),2))) + 1)
        #validating the number of clusters with Elbow test
        SSE = []
        K = range(1,options.elbow_clusters)
        for k in K:
            kmeanModel = KMeans(n_clusters=k).fit(fc_matrix)
            kmeanModel.fit(fc_matrix)
            SSE.append(sum(np.min(cdist(fc_matrix,
                    kmeanModel.cluster_centers_, 'euclidean'),
                    axis=1)))
    else:
        subplots = len(list(combinations(range(options.pca_num),2)))


    #plotting figures
    fig=plt.figure(figsize=(4*subplots, 4), tight_layout=True)
    for pca_i, (pca_x, pca_y) in enumerate(combinations(
            range(options.pca_num),2)):
        ax = plt.subplot(1, subplots, pca_i + 1)
        x = y_pca[:, pca_x]
        y = y_pca[:, pca_y]
        colors = [plt.cm.viridis(float(i)/(options.cluster_num-1))
                for i in range(options.cluster_num)]
        for i in range(1, options.cluster_num + 1):
            xi = [x[j] for j in range(len(x)) if sortkmeans[j] == i]
            yi = [y[j] for j in range(len(y)) if sortkmeans[j] == i]
            ax.scatter(xi, yi, c=(colors[i-1],),
                    label=stat_list[i-1], alpha=0.5)
        ax.legend()
        ax.set_xlabel('PC' + str(pca_x + 1), fontsize=12)
        ax.set_ylabel('PC' + str(pca_y + 1), fontsize=12)

    if options.elbow_clusters:
        ax = plt.subplot(1, subplots, subplots)
        ax.plot(K, SSE, '-ok', c='black', alpha=0.5)
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('SSE', fontsize=12)

    if not options.no_display:
        plt.show()
    fig.savefig(os.path.splitext(options.infile)[0] + ".chep_k" +
            str(options.cluster_num) + ".pdf")
