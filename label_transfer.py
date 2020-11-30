import numpy as np



def label_count(Y):
    label_dict={}
    for y in Y:
        if y in label_dict.keys():
            label_dict[y]+=1
        else:
            label_dict[u]=1
    #print(label_dict)
    return label_dict

def label_vector_form(Y,label_index_dict,num_label):
    y_array=np.zeros((len(Y),num_label))
    i=0
    for y in Y:
        index=label_index_dict[y]
        y_array[i,index]=1
        i+=1

    return y_array

def label_num_form(Y,label_index_dict):
    y_array=np.zeros(len(Y))
    i=0
    for y in Y:
        index=label_index_dict[y]
        y_array[i]=index
        i+=1
    return y_array


def multi_label_vector(Y,label_index_dict,num_label):
    y_array=np.zeros((len(Y),num_label))
    i=0
    for item in Y:
        for y in item:
            index=label_index_dict[y]
            y_array[i,index]=1
        i+=1
    return y_array
    
