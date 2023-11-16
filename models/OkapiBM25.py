#import math
#import sys
#import time
import metapy
#import pytoml
import numpy as np
from rank_bm25 import BM25Okapi


class OkapiBM25Model:
    def __init__(self):
        self.model =  metapy.index.OkapiBM25(2.5222222222222225,0.62589363,91.384843)
        #self.model = BM25Okapi(tokenized_corpus)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

"""
def load_ranker(cfg_file, i, j,k):
    
    Use this function to return the Ranker object to evaluate, 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index.
    
    #return metapy.index.OkapiBM25(2.5222222222222225,0.62589363,91.384843) #11
    #return metapy.index.OkapiBM25(i,j,k)
"""
"""
#For Okapi25 only
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    bestscore = 0.0 
    bestparam_i = 0.0  # Doc term smoothing k1 to be in a range of 0.5-2.0
    bestparam_j = 0.0  # Length normalization b needs to be between 0 and 1
    bestparam_k = 0.0  # Query term smoothing 
 
    params_i = np.linspace(0,3,100)
    params_j = np.linspace(0.5,1.0,100)
    params_k = np.linspace(0,700,100)

    for i in params_i:
        for j in params_j:
            for k in params_k:
                cfg = sys.argv[1]
                print('Building or loading index...')
                idx = metapy.index.make_inverted_index(cfg)
                ranker = load_ranker(cfg, i, j, k)
                ev = metapy.index.IREval(cfg)

                with open(cfg, 'r') as fin:
                    cfg_d = pytoml.load(fin)

                query_cfg = cfg_d['query-runner']
                if query_cfg is None:
                    print("query-runner table needed in {}".format(cfg))
                    sys.exit(1)

                start_time = time.time()
                top_k = 10
                query_path = query_cfg.get('query-path', 'queries.txt')
                query_start = query_cfg.get('query-id-start', 0)

                query = metapy.index.Document()
                ndcg = 0.0
                num_queries = 0

                print('Running queries')
                with open(query_path) as query_file:
                    for query_num, line in enumerate(query_file):
                        query.content(line.strip())
                        results = ranker.score(idx, query, top_k)
                        ndcg += ev.ndcg(results, query_start + query_num, top_k)
                        num_queries+=1
                    ndcg= ndcg / num_queries

                    #print ("x", x) 
                    #bestscore = max(bestscore, ndcg) 
                    print ("i="+ str(i) +"j="+ str(j) + "k=" + str(k)) 
                    print("NDCG@{}: {}".format(top_k, ndcg))
                    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
    print ("bestparam= i="+ str(bestparam_i) +"j="+ str(bestparam_j) + "k=" + str(bestparam_k))
    print ("bestscore="+ str(bestscore))
"""    