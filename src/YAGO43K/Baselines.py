
import numpy as np
import pandas
import argparse, pickle, re

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import  hamming_loss

from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from collections import Counter


def load_embedding_model(Embedding_Path):
    ConnectE_embedding = {}
    
    with open(Embedding_Path,'r') as inf:
        for line in inf:
            lisplit=line.split('\t')
            ConnectE_embedding[lisplit[0]]=eval(lisplit[1])
    return ConnectE_embedding

# load pretrained ConnectE embeddding for YAGO embedding.    
def load_data(PATH, Embedding_Path, top_types):
    # load pretrained embeddding, YAGO embedding.    
    entity_embeddings = load_embedding_model(Embedding_Path)

    ## Lookup Entity Types in YAGO43K:
    YAGO_ttrain = list(filter(None, [re.split('\t', i.strip('\n')) for i in open(PATH+'/YAGO43k_Entity_Type_train_clean.txt')]))
    YAGO_ttest = list(filter(None, [re.split('\t', i.strip('\n')) for i in open(PATH+'/YAGO43k_Entity_Type_test_clean.txt')]))
    YAGO_tvalid = list(filter(None, [re.split('\t', i.strip('\n')) for i in open(PATH+'/YAGO43k_Entity_Type_valid_clean.txt')]))

    YAGO_types=YAGO_ttrain+YAGO_ttest+YAGO_tvalid

    Types_df = pandas.DataFrame(YAGO_types)
    Types_df=Types_df.set_index([0]) 

    Entities_Groups = Types_df.groupby(0).agg(lambda x: list(x)) # entities with multiple types

    # filtered vectors 42k out of 123k entities from YAGO3-10 dataset... 
    embedding_filter={} 
    y_dict={}

    for ent in Entities_Groups.index.values:
        
        if ent in entity_embeddings:
            embedding_filter[ent]= entity_embeddings[ent] 
            y_dict[ent]= Entities_Groups.loc[[ent]][1][0]
  
    X, y= filter_entities_topTypes(embedding_filter, y_dict, top_types)

    return X, y


def filter_entities_topTypes(embedding_vec, y_dict, top_types):
    
    # flatten y_true (list of lists) into one list to count the most frequent types.
    y_true=list(y_dict.values())
    y_true_flatten=sum(y_true, [])

    # count top_types in YAGO dataset.
    top_types=[key for key, _ in Counter(y_true_flatten).most_common(top_types)]

    #filter embeddings for  top_types entities
    entity_embedding_filter={}
    y_true_filter={}

    for ent, ttype in y_dict.items(): # y_true is a multi-label types (list of list)
            
        for ent_tt in ttype: 
                
            if ent_tt in top_types:
                entity_embedding_filter[ent]= embedding_vec[ent] # get the emb vec for entity
                    
                if ent in y_true_filter:
                    y_true_filter[ent]+= [ent_tt]
                else:
                    y_true_filter[ent]= [ent_tt]
                        
    X_all=np.array(list(entity_embedding_filter.values()))
    return X_all, y_true


def evaluation_results(y_test, y_pred):
    #----------- Evaluation based on Precision, Recall, Accuracy and F1-score: -------#
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='samples')
    recall = metrics.recall_score(y_test, y_pred, average='samples')
    f1 = metrics.f1_score(y_test, y_pred, average='samples')
    Hloss= hamming_loss(y_test,  y_pred)
    print("Evaluation results (acc. prec. rec. F1 & Hloss)-- Examples-based Averaged:\n\n{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(accuracy, precision, recall, f1, Hloss))


def preprocess_labels(y_true_filter):
    label_encoder = preprocessing.MultiLabelBinarizer()
    y_encoded=label_encoder.fit_transform(list(y_true_filter.values()))
    labels = label_encoder.classes_.tolist()
    return y_encoded

def KNN_baseline(X_l, y_l, B_test, y_test):
    parameters = {'k': range(1,1), 's': [0.5, 0.7, 1.0]}
    clf = GridSearchCV(MLkNN(), parameters, n_jobs= -1)
    clf.fit(X_l, y_l)
    y_KNN=clf.predict(B_test)
    evaluation_results(y_test, y_KNN.toarray())   

def LogisticRegression_baseline(X_l, y_l, B_test, y_test):
    Lr_MCLF = BinaryRelevance(LogisticRegression(solver='liblinear'))
    Lr_MCLF.fit(X_l, y_l)
    y_lr=Lr_MCLF.predict(B_test)
    evaluation_results(y_test, y_lr.toarray())

def RandomForest_baseline(X_l, y_l, B_test, y_test):
    rf_MCLF = BinaryRelevance(RandomForestClassifier(random_state=seed))
    rf_MCLF.fit(X_l, y_l)
    y_rf=rf_MCLF.predict(B_test)
    evaluation_results(y_test, y_rf.toarray())    

def DNN_baseline(X_l, y_l, B_test, y_test, B_valid, y_valid):

    n_patience=3
    EPOCHS= 100
    BATCH_SIZE= 128

    DNN_model=keras.Sequential(name="DNNModel")
    DNN_model.add(layers.Dense(128, activation='relu', input_shape=(X_l.shape[1],)))
    DNN_model.add(layers.Dense(y_l.shape[1], activation='sigmoid'))
    DNN_model.summary()

    # Compile model
    DNN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train the DNN baseline
    early= EarlyStopping(monitor='val_loss', mode='min', patience=n_patience, restore_best_weights=True)
    DNN_model.fit(X_l, y_l, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early], validation_data=(B_valid, y_valid), verbose=0)

    # print evaluation of DNN
    DNN_pred=DNN_model.predict(B_test)

    DNN_pred[DNN_pred <= 0.5] = 0
    DNN_pred[DNN_pred > 0.5] = 1
    DNN_pred=DNN_pred.astype(np.int64)

    evaluation_results(y_test, DNN_pred)

    
def main():

    # hyper-parameters
    labeled_size= 0.01      # size of labeled data, in our experiments we try 0.01, 0.1, 0.2, 0.9
    seed= 42                # random_seed to reproduce our results
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--PATH', help='Path of datasets FB15k-ET or YAGO43k-ET.', default='./data/YAGO43K-ET/')
    parser.add_argument('--Embedding_Path', help='Path of the embedding model.', default='./data/YAGO43K-ET/Preprocessed Files/YAGO43K-ConnectE.txt')
    parser.add_argument('--top_types', help='Number of top types {3, 5, 10}.', default=10)
    args=parser.parse_args()

    # load the dataset
    X_all, y_all = load_data(args.PATH, args.Embedding_Path)

    # preprocess entities' labels
    y_encoded = preprocess_labels(y_all)

    # train-valid-test dataset split into label-unlabelled (x_,x_u), (y_l, y_u)
    X_l, X_u, y_l, y_u = train_test_split(X_all, y_encoded,  train_size=labeled_size, random_state=seed)
        
    # split the dataset B into test & valid sets
    X_valid, X_test, y_valid, y_test=train_test_split(X_u, y_u, train_size=labeled_size, random_state=seed)

    print ("Size of data: train-valid-test" , X_l.shape[0], X_valid.shape[0], X_test.shape[0])   

    print ('# Being the experiments: evaluate the baselines and save the best predictions\n')

    print ('# 1) evaluate the embedding baseline (KNN): \n')
    KNN_baseline(X_l, y_l, X_test, y_test)

    print ('# 2) evaluate logitistic regression baseline : \n')
    LogisticRegression_baseline(X_l, y_l, X_test, y_test)

    print ('# 3) evaluate randomForest baseline : \n')
    RandomForest_baseline(X_l, y_l, X_test, y_test)

    print ('# 4) evaluate DNN baseline : \n')
    teacher_pred=DNN_baseline(X_l, y_l, X_test, y_test, X_valid, y_valid)

    # save the predictions of best baseline. In our experiments, DNN shows as a strong baseline.
    PATH_teacherPred= args.PATH+'Preprocessed Files/y_DNN.pkl'
    outfile = open(PATH_teacherPred, 'wb') 
    pickle.dump(teacher_pred, outfile)

    print ('# END #')

if __name__ == "__main__":
    main()