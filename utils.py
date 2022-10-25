import pandas as pd
import numpy as np
def get_dict(file_name):
    file_c = pd.read_csv(file_name,delimiter=' ')
    dict={}

    for i in range(len(file_c)):
        k = file_c.loc[i][0]
        v = file_c.loc[i][1]
        dict[k] = v
    return dict

def cosine_similarity(vect_1,vect_2):
    cos=-10
    dot=np.dot(vect_1,vect_2)
    cos = dot/(np.linalg.norm(vect_1)*np.linalg.norm(vect_2))

    return cos