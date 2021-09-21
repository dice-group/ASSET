import re, pandas, pickle, argparse
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import   hamming_loss

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from collections import Counter


# For loading ConnectE embedding model For YAGO3-10 Dataset
def load_embedding_model(Embedding_Path):
    ConnectE_embedding = {}
    
    with open(Embedding_Path,'r') as inf:
        for line in inf:
            lisplit=line.split('\t')
            ConnectE_embedding[lisplit[0]]=eval(lisplit[1])
    return ConnectE_embedding

def load_dataset(PATH, Embedding_Path, top_types):
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

    # count top_types in FB15k dataset.
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


def preprocess_labels(y_true_filter):
    label_encoder = preprocessing.MultiLabelBinarizer()
    y_encoded=label_encoder.fit_transform(list(y_true_filter.values()))
    labels = label_encoder.classes_.tolist()
    return y_encoded

def evaluation_results(y_test, y_pred):
    #----------- Evaluation based on Precision, Recall, Accuracy and F1-score: -------#
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='samples')
    recall = metrics.recall_score(y_test, y_pred, average='samples')
    f1 = metrics.f1_score(y_test, y_pred, average='samples')
    Hloss= hamming_loss(y_test,  y_pred)
    print("Evaluation results (acc. prec. rec. F1 & Hloss)-- Examples-based Averaged:\n\n{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(accuracy, precision, recall, f1, Hloss))

def student_model(X, y, dropout_rate=0.25): 

    studentModel=keras.Sequential(name="StudentModel")
    studentModel.add(layers.Dense(128, activation='relu', input_shape=(X.shape[1],)))
    studentModel.add(layers.Dropout(rate=dropout_rate))
    studentModel.add(layers.Dense(y.shape[1], activation='sigmoid'))
    studentModel.summary()
    studentModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return studentModel

def train_model(student_model, x_l, x_u, x_valid, y_l, y_teacher,  y_valid, y_test): 
    # Train mode
    BATCH_SIZE = 128
    EPOCHS = 100
    n_patience=3
    n_iterations=5

    early= EarlyStopping(monitor='val_loss', mode='min', patience=n_patience, restore_best_weights=True)

    # get the pseudo-labels initially from the teacher model 
    teacher_pred=y_teacher  # initialize the pseudo-labels prediction with the initial teacher's predictions

    for i in range(n_iterations):

        # Train the Student Model on the Labeled Dataset:
        student_model.fit(x_l, y_l, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early], validation_data=(x_valid, y_valid), verbose=0)

        # Train the student model on the pseudo-labeled data:
        student_model.fit(x_u, teacher_pred, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early], validation_data=(x_valid, y_valid), verbose=0)

        # We use the DNN baseline as our initial teacher.
        teacher_pred=student_model.predict(y_test) 

    print ('# Training is done, now evaluation.#')
    teacher_pred[teacher_pred <= 0.5] = 0
    teacher_pred[teacher_pred > 0.5] = 1
    teacher_pred=teacher_pred.astype(np.int64)
    evaluation_results(y_test, teacher_pred)    

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
    X_all, y_all = load_dataset(args.PATH, args.Embedding_Path, args.top_types)

    # preprocess entities' labels
    y_encoded = preprocess_labels(y_all)

    # train-valid-test dataset split into label-unlabelled (x_,x_u), (y_l, y_u)
    x_l, x_u, y_l, y_u = train_test_split(X_all, y_encoded,  train_size=labeled_size, random_state=seed)
    
    # split the dataset B into test & valid sets
    x_valid, x_test, y_valid, y_test=train_test_split(x_u, y_u, train_size=labeled_size, random_state=seed)

    print ("Size of data: train-valid-test" ,   x_l.shape[0], x_valid.shape[0], x_test.shape[0])   

    #load predictions from the initial teacher model. (in our experiments, DNN shows the best predictions) 
    PATH_teacherPred= args.PATH+'Preprocessed Files/y_DNN.pkl'
    y_teacher= pickle.load(open(PATH_teacherPred, "rb"))

    
    # train our semi-supervisied model, and print the evaluation results.
    train_model(student_model, x_l, x_u, x_valid, y_l, y_teacher, y_u, y_valid, y_test)

if __name__ == "__main__":
    main()
