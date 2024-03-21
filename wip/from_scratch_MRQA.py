# MRQA python implementation
import numpy as np

def mrqa(data: np.array, embed_dims=1, delay=1, norm='euc', radius=1, zscore=0):
    parameters = {'embed_dims': embed_dims,
                    'delay': delay,
                    'norm': norm,
                    'radius': radius,
                    'zscore': zscore}
    
    dim = data.size()
    
    if embed_dims > 1: # if embed_dims > 1, perform time-delayed embbedding
        for i in range(embed_dims):
            _tempx = len(data)-(embed_dims-1)*delay -1
            _tempy =  : dim*i
            tempdata[:_tempx, 1+dim*i:dim*(i+1)] = data[range(1+(i-1)*delay, length(data)-(embed_dims-i)*delay),:]
        data=tempdata
        clear tempdata
    else


    return rp, results, parameters, b