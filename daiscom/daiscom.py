import sys
import time
import numpy as np
import pandas as pd
import argparse
import random
from random import randint, sample
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import function
#--------------------------------------        
#--------------------------------------        
# main

# read parameters
parser = argparse.ArgumentParser(description='DIASCOM deconvolution.')
parser.add_argument("-M", type=str, help="The mode of DAISCOM, [Os]: One-stop, [Op]: Only Prediction, [Tp]: Train and Prediction", default="Os")
parser.add_argument("-P", type=str, help="The platform of data, [Rc]: RNA-seq readcounts, [Rt]: RNA-seq tpm, [M]: Microarray", default="Rc")
parser.add_argument("-E", type=str, help="The reference real sample expression file/ simulated sample expression file", default="../data/testdata/example_refexp.txt")
parser.add_argument("-G", type=str, help="The reference real sample ground truth file/ simulated sample ground truth file", default="../data/testdata/example_reffra.txt")
parser.add_argument("-N", type=int, help="The number of training sample to simulate from each reference sample", default=400)
parser.add_argument("-F", type=str, help="The selected marker genes file", default="../data/feature/sig_genes395_sort.txt")
parser.add_argument("-I", type=str, help="The target sample expression file", default="../data/testdata/example_tarexp.txt")
parser.add_argument("-O", type=str, help="The directory of output result file", default="../output/")
parser.add_argument("-D", type=str, help="The deep-learing model  file trained by DAISCOM", default="../output/DAISCOM_model.pkl")
parser.add_argument("-C", type=str, help="The DAISCOM model celltypes file", default="../output/DAISCOM_celltypes.txt")
args = parser.parse_args()


# parameters
random_seed = 777
min_f = 0.01
max_f = 0.99
lr = 1e-4
batchsize = 32
num_epoches = 500
mode = args.M

##判断输入gene是否一致，样本数即排序是否一致，细胞类型命名是否正确，否则返回报错

if mode == "Os":
    
    platform = args.P
    refexp = pd.read_csv(args.E, sep="\t", index_col=0)
    trainfra = pd.read_csv(args.G, sep="\t", index_col=0)
    marker = pd.read_csv(args.F,sep='\t')
    target_sample = pd.read_csv(args.I, sep="\t", index_col=0)
    trainnum = args.N
    outdir = args.O
   
    (trainsam,testsam,commongene,C_B,C_NK,C_CD4,C_CD8,C_MONO,C_FIB,C_END,C_NEU) = function.daiscom_preprocessing(refexp,target_sample,marker,outdir,platform)   
    (mixsam, mixfra, celltypes) = function.daiscom_simulation(trainsam, trainfra,C_B,C_NK,C_CD4,C_CD8,C_MONO,C_FIB,C_END,C_NEU, random_seed, trainnum, outdir, min_f, max_f)
    model = function.daiscom_training(mixsam, mixfra, random_seed,outdir, num_epoches, lr, batchsize)
    result = function.daisocm_prediction(model, testsam, celltypes, commongene,outdir)
    print(result)

elif mode == "Op":
    
    modelpath = args.D
    marker = pd.read_csv(args.F,sep='\t')
    celltypes = pd.read_csv(args.C,sep='\t')
    target_sample = pd.read_csv(args.I, sep="\t", index_col=0)
    outdir = args.O
    
    commongene = list(marker["0"])
    celltypes = list(celltypes["0"])
    testsam = target_sample.reindex(commongene)
    
    model = function.model_load(commongene, celltypes, modelpath, outdir, random_seed)
    result = function.daisocm_prediction(model, testsam, celltypes, commongene,outdir)
    print(result)
    
elif mode == "Tp":
    
    refexp = pd.read_csv(args.E, sep="\t", index_col=0)
    mixfra = pd.read_csv(args.G, sep="\t", index_col=0)
    marker = pd.read_csv(args.F,sep='\t')
    target_sample = pd.read_csv(args.I, sep="\t", index_col=0)
    outdir = args.O
    
    commongene = list(set(list(marker["0"])).intersection(list(refexp.index)).intersection(list(target_sample.index)))
    commongene.sort()
    mixsam = refexp.reindex(commongene)
    testsam = target_sample.reindex(commongene)
    celltypes = list(mixfra.index)
    
    model = function.daiscom_training(mixsam, mixfra, random_seed,outdir, num_epoches, lr, batchsize)
    result = function.daisocm_prediction(model, testsam, celltypes, commongene, outdir)
    print(result)
    
else: 
    print("ERROR: unknown mode.")
    sys.exit(1)




