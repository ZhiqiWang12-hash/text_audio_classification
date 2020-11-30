import numpy as np

def categorical_accuracy(pred,y):
    correct=0
    confusion_matrix=np.zeros((pred.shape[1],pred.shape[1])).astype('int32')
    for i in range(pred.shape[0]):
        pred_index=np.argmax(pred[i])
        true_index=np.argmax(y[i])
        #print(pred_index,true_index)
        if pred_index==true_index:
            correct+=1
        confusion_matrix[pred_index,true_index]+=1

    acc=np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        acc[i]=confusion_matrix[i,i]/np.sum(confusion_matrix[:,i])
    performance_matrix={}
    performance_matrix['unweighted accuracy']=correct/pred.shape[0]
    performance_matrix['weighted accuracy']=np.mean(acc)
    performance_matrix['accuracy per class']=acc
    performance_matrix['confusion_matrix']=confusion_matrix
    return performance_matrix


def binary_multiclass_evaluation(pred,Y_test):
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pre=pred[i,j]
            true=Y_test[i,j]
            if true==1 and pre>=0.5:
                tp+=1
            elif true==0 and pre<0.5:
                tn+=1
            elif true==0 and pre>=0.5:
                fp+=1
            elif true==1 and pre<0.5:
                fn+=1

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    performance_matrix={}
    performance_matrix['accuracy']=(tp+tn)/(tp+tn+fp+fn)
    performance_matrix['recall']=tp/(tp+fn)
    performance_matrix['precision']=tp/(tp+fp)
    performance_matrix['score']=(2 * precision * recall) / (precision + recall)
    return performance_matrix

def binary_multiclass_evaluation_one_class(pred,Y_test,j):
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(pred.shape[0]):
        pre=pred[i,j]
        true=Y_test[i,j]
        if true==1 and pre>=0.5:
            tp+=1
        elif true==0 and pre<0.5:
            tn+=1
        elif true==0 and pre>=0.5:
            fp+=1
        elif true==1 and pre<0.5:
            fn+=1
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    performance_matrix={}
    performance_matrix['accuracy']=(tp+tn)/(tp+tn+fp+fn)
    performance_matrix['recall']=tp/(tp+fn)
    performance_matrix['precision']=tp/(tp+fp)
    performance_matrix['score']=(2 * precision * recall) / (precision + recall)
    return performance_matrix
