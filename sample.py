import six
assert six.PY3, "Run me with python3"
import sys
import json
import sklearn.neighbors
import numpy as np
import random

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

    def __init__(self,to_sample,stats_json):
        with open(stats_json,"rt") as f:
            probs=json.load(f)
        self.buckets=np.ceil(np.exp(probs)*np.double(to_sample))
        self.buckets[0]=0 #Do not sample length one
        self.buckets=self.buckets[:30] #don't bother with stuff > 100
        print(self.buckets,file=sys.stderr)

    def sample(self,tree):
        bucket_idx=len(tree)-1
        if bucket_idx>=len(self.buckets) or self.buckets[bucket_idx]==0:
            return False
        else:
            self.buckets[bucket_idx]-=1
            return True

    def report_missing(self):
        print("Length  --   Missing trees",file=sys.stderr)
        for l,miss in enumerate(self.buckets):
            print(l+1,"     ",miss,file=sys.stderr)
    
    @classmethod
    def estimate(cls,inp,max_trees=100000,save_json_file=None):
        lengths=[]
        for tree,comments in read_conll(inp,max_trees):
            lengths.append([len(tree)])
        estimator=sklearn.neighbors.KernelDensity(kernel="gaussian",bandwidth=2)
        estimator.fit(lengths)
        logprobs=list(estimator.score_samples([[i] for i in range(1,1000)])) #Sample probabilities in range 1-1000
        if save_json_file is not None:
            json.dump(logprobs,save_json_file)
#        print(*50000).astype(np.int))

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--estimate', type=int, default=0, metavar="NUMTREES", help='Estimate from stdin, dump json to stdout. Estimate based on NUMTREES trees.')
    parser.add_argument('--stats', dest='stats_json', default=None, action="store", help='Json file with the stats produced by --estimate')
    parser.add_argument('--sample', default=10000, action="store", help='Aim at sampling this many trees total. Default %(default)d')
    parser.add_argument('--max-trees', default=500000, action="store", help='Max number of trees to read when sampling. Default %(default)d')

    args = parser.parse_args()

    sampled_trees=[]
    
    if args.estimate>0:
        Stats.estimate(sys.stdin,max_trees=args.estimate,save_json_file=sys.stdout)
    elif args.stats_json is not None:
        stats=Stats(args.sample,args.stats_json)
        for tree,comments in read_conll(sys.stdin,0):
            if stats.sample(tree):
                sampled_trees.append((tree,comments))
        stats.report_missing()
        random.shuffle(sampled_trees)
        for tree,comments in sampled_trees:
            if comments:
                print("\n".join(comments))
            print("\n".join("\t".join(cols) for cols in tree))
            print()
            


            
                
