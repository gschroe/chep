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

CHEP_VERSION = "0.1.7"

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
    def __init__(self, infile):
        self.infile = infile
        self._var = None
        self._data = None

    def var(self):
        if self._var:
            return self._var
        else:
            self._parse_file()
            return self._var

    def data(self):
        if self._data:
            return self._data
        else:
            self._parse_file()
            return self._data

    def _parse_file(self):
        var = [] #read the variables
        data = [] # read the data

        for line in open(self.infile).readlines():
            if "_rln" in line:
                var.append(line.split()[0])
            elif ("data_" in line or
                    "loop_" in line or
                    len(line.strip()) == 0):
                continue
            else:
                data.append(line.split())
        self._var = var
        self._data = data


class WriteStar(object):
    def __init__(self, file_name, var, data):
        self.file_name = file_name
        self.var = var
        self.data = data

    def write(self):
        with open(self.file_name, "w") as outfile:
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

    if (len(rlnvar) ==
            len(rlndata_list[0])):
        rlndata = np.asarray(rlndata_list)
    else:
        print("ERROR: Number of metadata variables does not match " +
                "number of columns in the input STAR file")
        sys.exit(2)

    rlnvar_nr = len(rlnvar)
    print("Number of parameters in the relion file = " +
            str(int(rlnvar_nr)))

    ptcl_nr=len(rlndata_list)
    print("Number of particles in the relion file = " +
            str(int(ptcl_nr)))

    rlntubeID_col = rlnvar.index('_rlnHelicalTubeID')
    rlnclassID_col = rlnvar.index('_rlnClassNumber')
    rlnMicrographName_col = rlnvar.index('_rlnMicrographName')

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

        cluster_out = WriteStar(file_name, rlnvar, cluster_data)
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
