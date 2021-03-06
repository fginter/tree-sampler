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

def read_conll(inp,maxsent,args):
    """ Read conll format file and yield one sentence at a time as a list of lists of columns. If inp is a string it will be interpreted as filename, otherwise as open file for reading in unicode"""
    count=0
    sent=[]
    comments=[]
    seen=set() #set of trees seen so far
    dups=0
    for line in inp:
        line=line.strip()
        if not line:
            if sent:
                if len(sent)<args.min_tree_len:
                    sent=[]
                    comments=[]
                    continue
                if args.dedup:
                    txt="".join(cols[1] for cols in sent).lower()
                    if txt in seen:
                        #print(" ".join(cols[1] for cols in sent).lower(),file=sys.stderr)
                        dups+=1
                        sent=[]
                        comments=[]
                        continue
                    else:
                        seen.add(txt)
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
            if args.dedup:
                txt="".join(cols[1] for cols in sent).lower()
                if txt in seen:
                    dups+=1
                else:
                    yield sent, comments
            elif len(sent)>=args.min_tree_len:
                yield sent, comments
    if args.dedup:
        print("Removed {} duplicates".format(dups),file=sys.stderr)

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
    def hist_estimate(cls,target_features,args):
        tree_len_bins=np.arange(1,30,2)
        dtype_bins=np.arange(1,30,2)
        target_H,xedge,yedge=np.histogram2d(target_features[:,0],target_features[:,1],bins=[tree_len_bins, dtype_bins],normed=False)
        target_H+=1#0.01*np.sum(target_H)/(len(tree_len_bins)*len(dtype_bins)) #redistribute 1% of mass to unseen 
        target_H/=np.sum(target_H)

        # sampling=(np.log(target_H)-np.log(source_H))-np.log(args.divider) #divide by four to get better shot at the long ones
        # sampling=np.power(np.clip(np.exp(sampling),a_min=None,a_max=1.0),args.power) #sampling is now the probabilities with which to sample the trees
        # if args.visualize:
        #     import matplotlib.pyplot as plt
        #     plt.pcolor(xedge,yedge,sampling.T,vmin=0,vmax=1,cmap="gray_r")
        #     plt.colorbar()
        #     plt.title("Sampling of pure data (white downsampled)")
        #     plt.xlabel("Tree length")
        #     plt.ylabel("Num dep types")
        #     plt.show()
        return (tree_len_bins,dtype_bins), target_H

        
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

def sample(count_table,src_trees,bins,indices,to_sample):
    sampled=0
    for i in range(len(indices)):
        if indices[i]==-1:
            continue #seen,skip
        idx=indices[i]
        tlen_bin,dtype_bin=bins[idx]
        if count_table[tlen_bin,dtype_bin]>1: #still some to go
            count_table[tlen_bin,dtype_bin]-=1
            indices[i]=-1
            sampled+=1
            tree,comments=src_trees[idx]
            if comments:
                print("\n".join(comments))
            print("\n".join("\t".join(cols) for cols in tree))
            print()
            if sampled>=to_sample: #we're done
                break
    return sampled


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--dedup',default=False, action="store_true",help="Dedup the input before doing anything else")
    parser.add_argument('--estimate-src-max', type=int, default=20000, metavar="NUMTREES", help='Estimate from a maximum of this many trees. The more the better. Default %(default)d')
    parser.add_argument('--estimate-tgt-max', type=int, default=20000, metavar="NUMTREES", help='Estimate target based on NUMTREES trees. Default %(default)d')
    parser.add_argument('--estimate-tgt-trees', default=None, help='Use trees from this file as target distribution data.')
    parser.add_argument('--min-tree-len',default=0,type=int,help="Minimum length of a tree to consider. Default %(default)d")
#    parser.add_argument('--visualize',default=False,action="store_true",help="Plot the sampling probabilities")
#    parser.add_argument('--divider',default=8.0,type=float,help="Constant divider for the probabilities. The higher the number, the less data is sampled, but the more fidelity. Default %(default)f")
#    parser.add_argument('--power',default=1.0,type=float,help="Take power of the sampling probabilities, further increasing chance that rare trees get sample. Default %(default)f")
    parser.add_argument('--max-output',default=0,type=int,help="Sample max this many trees. Zero means all. Default %(default)d")
    parser.add_argument('--tok',default=False,action="store_true",help="--max-output is counted as tokens, not trees. Only works with --shuffle, sorry.")
    parser.add_argument('--random',type=float,help="Sample with a random rate given as number between 0-1. Stop when max-output reached.")
    parser.add_argument('--shuffle',default=False, action="store_true",help="Read input, shuffle, sample max-output trees.")
    parser.add_argument('--orphan',default=False, action="store_true",help="Only pass trees with orphans.")
    
    

    args = parser.parse_args()

    sampled=0
    total=0
    if args.random is not None:
        for tree,comments in read_conll(sys.stdin,0,args):
            total+=1
            if random.random()<args.random:
                sampled+=1
                if comments:
                    print("\n".join(comments))
                print("\n".join("\t".join(cols) for cols in tree))
                print()
                if args.max_output>0 and sampled>=args.max_output:
                    break
    elif args.orphan:
        for tree,comments in read_conll(sys.stdin,0,args):
            total+=1
            all_deps=set(cols[DEPREL] for cols in tree)
            if "orphan" not in all_deps:
                continue
            else:
                sampled+=1
                if comments:
                    print("\n".join(comments))
                print("\n".join("\t".join(cols) for cols in tree))
                print()
                if args.max_output>0 and sampled>=args.max_output:
                    break
    elif args.shuffle:
        trees=list(read_conll(sys.stdin,0,args))
        random.shuffle(trees)
        total=len(trees)
        if args.max_output==0: #zero means all as the help says...
            src_trees=trees
        else:
            if args.tok:
                src_trees=[]
                sampled_tokens=0
                for (tree,comments) in trees:
                    sampled_tokens+=len(tree)
                    src_trees.append((tree,comments))
                    if sampled_tokens>=args.max_output:
                        break
                print("Sampled {} tokens in {} trees, target was {}".format(sum(len(x[0]) for x in src_trees),len(src_trees),args.max_output),file=sys.stderr)
            else:
                src_trees=itertools.islice(trees,args.max_output)
        for tree,comments in src_trees:
            sampled+=1
            if comments:
                print("\n".join(comments))
            print("\n".join("\t".join(cols) for cols in tree))
            print()
    elif args.estimate_tgt_trees is not None: #we are estimating density and sampling
        print("Estimation",file=sys.stderr)
        if args.estimate_tgt_trees.endswith(".gz"):
            inp=gzip.open(args.estimate_tgt_trees,"rt")
        else:
            inp=open(args.estimate_tgt_trees,"rt")
        #read training data
        target_trees=list(read_conll(inp,args.estimate_tgt_max,args))
        target_features=Stats.tree_features(target_trees)
        (tree_len_bins,dtype_bins),prob_table=Stats.hist_estimate(target_features,args)
        target_table=prob_table*args.max_output #how much do we want of what?
        count_table=target_table.copy()
        #print("Count table",count_table,file=sys.stderr)
        
        print("Reading input",file=sys.stderr)
        one_batch=list(read_conll(sys.stdin,args.estimate_src_max,args))
        print("Constructing bins",file=sys.stderr)
        bins=[]
        for tree,comments in one_batch:
            feats=Stats.tree_features([(tree,comments)])[0]
            tlen_bin=np.digitize(feats[0],tree_len_bins)
            dtype_bin=np.digitize(feats[1],dtype_bins)
            tlen_bin=tlen_bin.clip(None,count_table.shape[0]-1)
            dtype_bin=dtype_bin.clip(None,count_table.shape[1]-1)
            bins.append((tlen_bin,dtype_bin))
        indices=np.arange(len(one_batch))
        np.random.shuffle(indices)

        print("Read and shuffled {} trees".format(len(one_batch)),file=sys.stderr)
        total_sampled=0
        min_to_get=args.max_output
        while True:
            count_table=prob_table*(min_to_get)
            sampled=sample(count_table,one_batch,bins,indices,args.max_output-total_sampled)
            if sampled==0: #we're screwed, nothing can be done anymore
                print("Cannot sample any more, quitting",file=sys.stderr)
                break
            total_sampled+=sampled
            print("Sampled in round: {} for a total {} out of {}".format(sampled,total_sampled,args.max_output),file=sys.stderr)
            if total_sampled>=args.max_output:
                break
            #How much is missing?
            missing=args.max_output-total_sampled
            min_to_get=max(5.1/np.max(prob_table),missing*1.2)
            print("New min to get",min_to_get,file=sys.stderr)
            sampled=total_sampled
    else:
        print("Don't know what to do. You need to give --estimate-* or --shuffle or --random.",file=sys.stderr)
        sys.exit(-1)

    print("Sampled {} of {} trees".format(sampled,total),file=sys.stderr)

