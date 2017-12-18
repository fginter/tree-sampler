import six
assert six.PY3, "Run me with python3"
import gzip
import sys
import json
import sklearn.neighbors
import sklearn.preprocessing
import numpy as np
import random
import pickle
import itertools
import scipy

ID,FORM,LEMMA,UPOS,XPOS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)

def read_conll(inp,maxsent):
    """ Read conll format file and yield one sentence at a time as a list of lists of columns. If inp is a string it will be interpreted as filename, otherwise as open file for reading in unicode"""
    count=0
    sent=[]
    comments=[]
    for line in inp:
        line=line.strip()
        if not line:
            if sent:
                count+=1
                yield sent, comments
                if maxsent!=0 and count>=maxsent:
                    break
                sent=[]
                comments=[]
        elif line.startswith("#"):
            if sent:
                raise ValueError("Missing newline after sentence")
            comments.append(line)
            continue
        else:
            sent.append(line.split("\t"))
    else:
        if sent:
            yield sent, comments


class Stats(object):

    
        
    @classmethod
    def tree_features(cls,trees):
        features=[] #tree features, right now (length, unique types per token), can be more in the future
        for tree,comments in trees:
            uniq_types=len(set(cols[DEPREL] for cols in tree))
            features.append([len(tree),uniq_types])
        features=np.asarray(features,dtype=np.double)
        return features

    @classmethod
    def hist_estimate(cls,source_features,target_features,args):
        tree_len_bins=np.arange(1,30,2)
        dtype_bins=np.arange(1,30,2)
        source_H,xedge,yedge=np.histogram2d(source_features[:,0],source_features[:,1],bins=[tree_len_bins, dtype_bins],normed=False)
        target_H,xedge,yedge=np.histogram2d(target_features[:,0],target_features[:,1],bins=[tree_len_bins, dtype_bins],normed=False)
        source_H+=0.01*np.sum(source_H)
        source_H/=np.sum(source_H)
        target_H+=0.01*np.sum(target_H)
        target_H/=np.sum(target_H)
        sampling=(np.log(target_H)-np.log(source_H))-np.log(args.divider) #divide by four to get better shot at the long ones
        sampling=np.power(np.clip(np.exp(sampling),a_min=None,a_max=1.0),args.power) #sampling is now the probabilities with which to sample the trees
        if args.visualize:
            import matplotlib.pyplot as plt
            plt.pcolor(xedge,yedge,sampling.T,vmin=0,vmax=1,cmap="gray_r")
            plt.colorbar()
            plt.title("Sampling of pure data (blue downsampled)")
            plt.xlabel("Tree length")
            plt.ylabel("Num dep types")
            plt.show()
        return (tree_len_bins,dtype_bins), sampling

    @classmethod
    def grid(cls,scaler,estimator):
        lengths=np.arange(1,30,1)
        uniq_types=np.arange(0,1,0.05)
        features=np.stack((np.meshgrid(lengths,uniq_types)),-1).reshape(-1,2)
        norm_features=scaler.transform(features)
        log_scores=estimator.score_samples(norm_features)
        norm_const=scipy.special.logsumexp(log_scores)
        log_scores-=norm_const
        return features,log_scores
        
    @classmethod
    def sample(cls,tree,bins,sampling_table):
        tree_len_bins,dtype_bins=bins
        feats=cls.tree_features([tree])[0]
        tlen_bin=np.digitize(feats[0],tree_len_bins)
        dtype_bin=np.digitize(feats[1],dtype_bins)
        tlen_bin=tlen_bin.clip(None,sampling_table.shape[0]-1)
        dtype_bin=dtype_bin.clip(None,sampling_table.shape[1]-1)
        sampling_prob=sampling_table[tlen_bin,dtype_bin]
        #print("Feats",feats,"sample prob",sampling_prob)
        return np.random.random()<sampling_prob

        

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--estimate-src-max', type=int, default=20000, metavar="NUMTREES", help='Estimate source based on NUMTREES trees. Default %(default)d')
    parser.add_argument('--estimate-src-trees', default=None, help='Use trees from this file as source distribution data.')
    parser.add_argument('--estimate-tgt-max', type=int, default=20000, metavar="NUMTREES", help='Estimate target based on NUMTREES trees. Default %(default)d')
    parser.add_argument('--estimate-tgt-trees', default=None, help='Use trees from this file as target distribution data.')
    parser.add_argument('--visualize',default=False,action="store_true",help="Plot the sampling probabilities")
    parser.add_argument('--divider',default=4.0,type=float,help="Constant divider for the probabilities. The higher the number, the less data is sampled, but the more fidelity. Default %(default)f")
    parser.add_argument('--power',default=2.0,type=float,help="Take power of the sampling probabilities, further increasing chance that rare trees get sample. Default %(default)f")
    parser.add_argument('--max-output',default=50000,type=int,help="Sample max this many trees. Default %(default)f")

    args = parser.parse_args()

    if args.estimate_src_trees is not None and args.estimate_tgt_trees is not None: #we are estimating density
        print("Estimation",file=sys.stderr)
        if args.estimate_src_trees.endswith(".gz"):
            inp=gzip.open(args.estimate_src_trees,"rt")
        else:
            inp=open(args.estimate_src_trees,"rt")
        source_trees=list(read_conll(inp,args.estimate_src_max))

        if args.estimate_tgt_trees.endswith(".gz"):
            inp=gzip.open(args.estimate_tgt_trees,"rt")
        else:
            inp=open(args.estimate_tgt_trees,"rt")
        #read training data
        target_trees=list(read_conll(inp,args.estimate_tgt_max))

        source_features=Stats.tree_features(source_trees)
        target_features=Stats.tree_features(target_trees)
        bins,sampling_table=Stats.hist_estimate(source_features,target_features,args)
    else:
        print("You need to provide --estimate-...-trees parameters",file=sys.stderr)
        sys.exit(-1)

    #and now sample
    sampled=0
    total=0
    for tree,comments in read_conll(sys.stdin,0):
        total+=1
        if Stats.sample((tree,comments),bins,sampling_table):
            sampled+=1
            if comments:
                print("\n".join(comments))
            print("\n".join("\t".join(cols) for cols in tree))
            print()
            if sampled>=args.max_output:
                break
    print("Sampled {} of {} trees".format(sampled,total),file=sys.stderr)

            


            
                
