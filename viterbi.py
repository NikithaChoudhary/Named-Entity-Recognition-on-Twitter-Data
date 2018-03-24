import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is a size N array of integers representing the best sequence.
    """

    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    trellis = [[0 for i in range(N)] for t in range(L)]
    back = [[0 for i in range(N)] for t in range(L)]
    for t in xrange(L): 
        trellis[t][0]= start_scores[t] + emission_scores[0][t]
    for i in xrange(1,N):
        for t in xrange(L):
            trellis[t][i] =  -np.inf
            for t_p in xrange(L):
                tmp = trellis[t_p][i-1] + trans_scores[t_p][t]
                if(tmp > trellis[t][i]) :
                    trellis[t][i] = tmp
                    back[t][i] = t_p
            trellis[t][i] += emission_scores[i][t]

    t_max = 0
    vit_max = -np.inf

    for t in xrange(L):
        if(trellis[t][N-1] + end_scores[t] > vit_max):
            t_max = t
            vit_max= trellis[t][N-1]+ end_scores[t]
    ind = N-1
    tags = []#new array[n+1]
    t = t_max
    while(ind >= 0):
        tags = [t] + tags #tags.insert(0,t)
        t = back[t][ind]
        ind = ind -1   
 
    return (vit_max, tags)
