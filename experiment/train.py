#python train.py --model,face_fusion,audio_feature,emotion_feature --epochs 300 --task emotion --fusion --fusion_type early
#python train.py --model emotion_feature --epochs 100 --task emotion

import sys
import glob
import tensorflow
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from models import ResearchModels
from data import DataSet
import tensorflow as tf
import time
import os.path
import pdb
import functions
import numpy as np
import argparse
sys.path.append('..')
from calculateEvaluationCCC import ccc, mse, f1
from utils import  display_true_vs_pred, print_out_csv, plot_acc, pad_tensor, padding
from subprocess import call
import subprocess
# define hyperparameters
FLAGS = tf.compat.v1.flags.FLAGS
#tf.app.flags.DEFINE_string("model", "trimodal_model", "the chosen model,should be one of : visual_model, audio_model, word_model, bimodal_model, trimodal_model.")
tf.compat.v1.flags.DEFINE_string("pretrained_model_path",None, "the pretrained_model_path.")
tf.compat.v1.flags.DEFINE_boolean("is_train", True, "True for training, False for evaluation.")
tf.compat.v1.flags.DEFINE_float("learning_rate", 1e-3, "The initial learning rate.")
tf.compat.v1.flags.DEFINE_integer("batch_size", 300, "The number of utterances in each batch.")

#for reproductivity
np.random.seed(123)
def load_custom_model(pretrained_model_path):
    model = load_model(pretrained_model_path, 
                           custom_objects ={'ccc_metric':functions.ccc_metric,
                                            'ccc_loss': functions.ccc_loss})
    return model

def train(istrain=True, model_type='trimodal', saved_model_path=None, task='emotion',
         batch_size=2, nb_epoch=200, learning_r=1e-3, show_plots=True, is_fusion=False,
         fusion_type=None, pretrained=False):
    """
    train the model
    :param model: 'visual_model','audio_model','word_model','trimodal_model'
    :param saved_model_path: saved_model path
    :param task: 'aoursal','valence','emotion'
    :param batch_size: 2
    :param nb_epoch:2100
    :return:s
    """
    timestamp =  time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
    # Helper: Save the model.
    model_name = model_type
    model_name = model_name.replace(':','-')
    model_name = model_name.replace('[','')
    model_name = model_name.replace(']','')
    if ',' in model_name:
        model_name = model_name.replace(',','__')
        max_len = 200
        if len(model_name) >= max_len:
            model_name = model_name[:max_len]
        model_name = 'fusion_' + fusion_type + '__' + model_name
    if not os.path.exists(os.path.join('checkpoints', model_name)):
        os.makedirs(os.path.join('checkpoints', model_name))
    checkpointer = ModelCheckpoint(
        monitor='val_accuracy',
        #filepath = os.path.join('checkpoints', model, task+'-'+ str(timestamp)+'-'+'best.hdf5' ),
        filepath = os.path.join('checkpoints', model_name, task + "-{val_accuracy:.3f}-{accuracy:.3f}.hdf5" ),
        verbose=1,
        save_best_only=True)
    checkpointer_acc = ModelCheckpoint(
        monitor='accuracy',
        #filepath = os.path.join('checkpoints', model, task+'-'+ str(timestamp)+'-'+'best.hdf5' ),
        filepath = os.path.join('checkpoints', model_name, task + "-{val_accuracy:.3f}-{accuracy:.3f}.hdf5" ),
        verbose=1,
        save_best_only=True)
    
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('logs', model_name))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=1000)
    
    # Helper: Save results.
    
    csv_logger = CSVLogger(os.path.join('logs', model_name , task +'-'+ \
        str(timestamp) + '.log'))

    # Get the data and process it.
    # seq_length for the sentence
    seq_length = 20
    dataset = DataSet(
        istrain = istrain,
        model = model_type,
        task = task,
        seq_length=seq_length,
        model_name=model_name,
        is_fusion=is_fusion
        )

    # Get the model.
    model = None
    if pretrained:
        model_weights_path = get_best_model(model_name)
        if model_weights_path:
            print('USING MODEL', model_weights_path)
            model = load_model(model_weights_path)
    if model is None:
        rm = ResearchModels(
                istrain = istrain,
                model = model_type, 
                seq_length = seq_length, 
                saved_path=saved_model_path, 
                task_type= task,
                learning_r = learning_r,
                model_name=model_name,
                is_fusion=is_fusion,
                fusion_type=fusion_type
                )
        model = rm.model
    # Get training and validation data.
    x_train, y_train, train_name_list = dataset.get_all_sequences_in_memory('Train')
    x_valid, y_valid, valid_name_list= dataset.get_all_sequences_in_memory('Validation')
    x_test, y_test, test_name_list = dataset.get_all_sequences_in_memory('Test')

    pdb.set_trace()

    if model_type == 'word_feature':
        x_train_max = max([i.shape[0] for i in x_train])
        x_test_max = max([i.shape[0] for i in x_test])
        x_valid_max = max([i.shape[0] for i in x_valid])

        maximus = max(x_train_max, x_test_max, x_valid_max)

        x_train = pad_tensor(x_train, maximus)
        x_test = pad_tensor(x_test, maximus)
        x_valid = pad_tensor(x_valid, maximus)

        pdb.set_trace()
        rm = ResearchModels(
                istrain = istrain,
                model = model_type, 
                seq_length = maximus, 
                saved_path=saved_model_path, 
                task_type= task,
                learning_r = learning_r,
                model_name=model_name,
                is_fusion=is_fusion,
                fusion_type=fusion_type
                )
        model = rm.model
    

    if task == 'emotion':
        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)
        y_test = to_categorical(y_test)
    # Fit!
    # Use standard fit
    print('Size', len(x_train), len(y_train), len(x_valid), len(y_valid), len(x_test), len(y_test))

    if 'audio_rnn' in model_type:
        x_train = np.reshape(x_train, (1, x_train.shape[0], x_train.shape[1]))
        y_train = np.reshape(y_train, (1, y_train.shape[0]))
        x_valid = np.reshape(x_valid, (1, x_valid.shape[0], x_valid.shape[1]))
        y_valid = np.reshape(y_valid, (1, y_valid.shape[0]))
        x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))
        y_test = np.reshape(y_test, (1, y_test.shape[0]))


    if 'audio_feature' in model_type:
        pdb.set_trace()
        x_train_reshaped = []
        x_test_reshaped = []
        x_valid_reshaped = []
        
        max_dim = max(x_test.shape[3], x_valid.shape[3], x_test.shape[3], x_train.shape[3], x_valid.shape[3], x_test.shape[3], x_valid.shape[3], x_valid.shape[3], x_valid.shape[3])
        for b_i in range(x_train.shape[0]):
            # max_dim = max(x_train.shape[3], x_valid.shape[3], x_test.shape[3])
            x_tr_mel = np.reshape(padding(x_train[b_i][0], x_train[b_i][0].shape[0], max_dim), (x_train[b_i][0].shape[0], max_dim, 1))
            x_tr_centr = np.reshape(padding(x_train[b_i][1], x_train[b_i][1].shape[0], max_dim), (x_train[b_i][1].shape[0], max_dim, 1))
            x_tr_bdwdth = np.reshape(padding(x_train[b_i][2], x_train[b_i][2].shape[0], max_dim), (x_train[b_i][2].shape[0], max_dim, 1))
            x_train_reshaped.append(np.concatenate((x_tr_mel, x_tr_centr, x_tr_bdwdth), axis=2))

        for b_i in range(x_test.shape[0]):
            # max_dim = max(x_test.shape[3], x_valid.shape[3], x_test.shape[3])
            x_tr_mel = np.reshape(padding(x_test[b_i][0], x_test[b_i][0].shape[0], max_dim), (x_test[b_i][0].shape[0], max_dim, 1))
            x_tr_centr = np.reshape(padding(x_test[b_i][1], x_test[b_i][1].shape[0], max_dim), (x_test[b_i][1].shape[0], max_dim, 1))
            x_tr_bdwdth = np.reshape(padding(x_test[b_i][2], x_test[b_i][2].shape[0], max_dim), (x_test[b_i][2].shape[0], max_dim, 1))
            x_test_reshaped.append(np.concatenate((x_tr_mel, x_tr_centr, x_tr_bdwdth), axis=2))

        for b_i in range(x_valid.shape[0]):
            # max_dim = max(x_valid.shape[3], x_valid.shape[3], x_valid.shape[3])
            x_tr_mel = np.reshape(padding(x_valid[b_i][0], x_valid[b_i][0].shape[0], max_dim), (x_valid[b_i][0].shape[0], max_dim, 1))
            x_tr_centr = np.reshape(padding(x_valid[b_i][1], x_valid[b_i][1].shape[0], max_dim), (x_valid[b_i][1].shape[0], max_dim, 1))
            x_tr_bdwdth = np.reshape(padding(x_valid[b_i][2], x_valid[b_i][2].shape[0], max_dim), (x_valid[b_i][2].shape[0], max_dim, 1))
            x_valid_reshaped.append(np.concatenate((x_tr_mel, x_tr_centr, x_tr_bdwdth), axis=2))
        
        x_train = np.asarray(x_train_reshaped)
        x_valid = np.asarray(x_valid_reshaped)
        x_test = np.asarray(x_test_reshaped)

    pdb.set_trace()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_valid,y_valid),
        verbose=1,
        callbacks=[tb, csv_logger,  checkpointer, checkpointer_acc],
        #callbacks=[tb, early_stopper, csv_logger,  checkpointer],
        #callbacks=[tb, lrate, csv_logger,  checkpointer],
        epochs=nb_epoch)
    
    # find the current best model and get its prediction on validation set
    model_weights_path = get_best_model(model_name)
    #model_weights_path = os.path.join('checkpoints', model_name, task + '-' + str(nb_epoch) + '-' + str(timestamp) + '-' + 'best.hdf5' )
    print('model_weights_path', model_weights_path)

    # if model_weights_path:
    #     best_model = load_custom_model(model_weights_path)
    # else:
    best_model = model
    

    y_valid_pred = best_model.predict(x_valid)
    y_valid_pred = np.squeeze(y_valid_pred)
    
    y_train_pred = best_model.predict(x_train)
    y_train_pred = np.squeeze(y_train_pred)

    y_test_pred = best_model.predict(x_test)
    y_test_pred = np.squeeze(y_test_pred)

    #calculate the ccc and mse

    if not os.path.exists('results'):
        os.mkdir('results')
    pdb.set_trace()
    filename = os.path.join('results', model_name+'__'+str(nb_epoch)+'_'+task+'.txt')
    # f1_score = f1(y_valid, y_valid_pred)
    # f1_score_test = f1(y_test, y_test_pred)
    acc_val = model.evaluate(x_valid, y_valid, verbose=1)[1]
    acc_train = model.evaluate(x_train, y_train, verbose=1)[1]
    acc_test = model.evaluate(x_test, y_test, verbose=1)[1]
    # print("F1 score in validation set is {}".format(f1_score))
    # print("F1 score in test set is {}".format(f1_score_test))
    print("Val acc is {}".format(acc_val))
    print("Train acc is {}".format(acc_train))
    print("Test acc is {}".format(acc_test))
    # plot_acc(history, model_name, timestamp, show_plots, nb_epoch)
    with open(filename, 'w') as f:
        f.write(str([acc_val, acc_train, acc_test]))
    # display the prediction and true label
    log_path = os.path.join('logs', model_name , task +'-'+ \
        str(timestamp) + '.log')
    
    # display_true_vs_pred([y_valid, y_train, y_test], [y_valid_pred, y_train_pred, y_test_pred],log_path, task, model_name, [acc_val, acc_train, acc_test], show_plots, timestamp, nb_epoch)

def get_best_model(model_name):
    best_model = None
    model_path = os.path.join('checkpoints', model_name)
    if os.path.exists(model_path):
        files = glob.glob(os.path.join(model_path, '*'))
        if len(files) > 0:
            files.sort()
            best_model = files[-1]
    return best_model

def evaluate_on_test(arousal_model_path, valence_model_path, output_file, istrain=False, model='trimodal'):
    arousal_model = load_custom_model(arousal_model_path)
    valence_model = load_custom_model(valence_model_path)
    dataset = DataSet(
        istrain = istrain,
        model = model
        )
    
    #load test data
    x_test, name_list = dataset.get_all_sequences_in_memory('Test')

    arousal_pred = arousal_model.predict(x_test)
    arousal_pred = np.squeeze(arousal_pred)
    valence_pred = valence_model.predict(x_test)
    valence_pred = np.squeeze(valence_pred)
    
    print_out_csv(arousal_pred, valence_pred, name_list, '../omg_TestVideos.csv', output_file)
    
def evaluate_on_validation(arousal_model_path, valence_model_path, output_file,istrain=True, model='trimodal'):
    arousal_model = load_custom_model(arousal_model_path)
    valence_model = load_custom_model(valence_model_path)
    dataset = DataSet(
    istrain = istrain,
    model = model,
    )
    x_valid, y_valid, valid_name_list= dataset.get_all_sequences_in_memory('Validation')
    
    arousal_pred = arousal_model.predict(x_valid)
    arousal_pred = np.squeeze(arousal_pred)
    valence_pred = valence_model.predict(x_valid)
    valence_pred = np.squeeze(valence_pred)
    
    print_out_csv(arousal_pred, valence_pred, valid_name_list, '../omg_ValidationVideos.csv', output_file)
    
    cmd = 'python ../calculateEvaluationCCC.py ../omg_ValidationVideos_pred.csv ../new_omg_ValidationVideos.csv'
    process = subprocess.Popen(cmd.split(),stderr= subprocess.STDOUT,universal_newlines=True)
    process.communicate()

def main():
    # pdb.set_trace()
    if FLAGS.is_train:
        show_plots = True
        pretrained = FLAGS.pretrained
        if ',' in FLAGS.task:
            tasks = FLAGS.task.split(',')
        else:
            if FLAGS.task == 'all':
                show_plots = False
                tasks = ['emotion', 'arousal', 'valence']
            else:
                tasks = [FLAGS.task]
        if FLAGS.fusion:
            if FLAGS.fusion_type == 'all':
                show_plots = False
                fusion_types = ['early', 'late']
            else:
                fusion_types = [FLAGS.fusion_type]
            if ';' in FLAGS.model:
                show_plots = False
                models = FLAGS.model.split(';')
            else:
                models = [FLAGS.model]
            for model in models:
                submodels = model.split(',')
                if len(submodels) > 1:
                    for fusion_type in fusion_types:
                        for task in tasks:
                            train(istrain=FLAGS.is_train, model_type = model, saved_model_path=FLAGS.pretrained_model_path, task = task,
                            batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epochs, learning_r = FLAGS.learning_rate, show_plots=show_plots,
                            is_fusion=True, fusion_type=fusion_type, pretrained=pretrained)
                else:
                    print('Fusion models must have at least two models separated by comma')
        else:
            if ',' in FLAGS.model:
                models = FLAGS.model.split(',')
            else:
                if FLAGS.model == 'all':
                    models = ['audio_feature', 'audio_rnn', 'emotion_feature', 'face_feature', 'face_visual', 'face_fusion', 'word_feature', 'word_mpqa', 'word_fusion', 'trimodal']
                else:
                    models = [FLAGS.model]
            if len(tasks) > 1:
                show_plots = False
            if len(models) > 1:
                show_plots = False
            for model in models:
                for task in tasks:
                    print(model, task)
                    train(istrain=FLAGS.is_train, model_type = model, saved_model_path=FLAGS.pretrained_model_path, task = task,
                        batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epochs, learning_r = FLAGS.learning_rate, show_plots=show_plots,
                        is_fusion=False, pretrained=pretrained)
        """
        arousal_model_path = ''
        valence_model_path = ''
        output_file = '../omg_ValidationVideos_pred.csv'
        evaluate_on_validation(arousal_model_path, valence_model_path, output_file,istrain=True)
        """
    else:
        arousal_model_path = ''
        valence_model_path = ''
        output_file = ''
        evaluate_on_test(arousal_model_path, valence_model_path, output_file,istrain=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is used to train a multimodal emotion recognition model.')
    parser.add_argument('--model', default='trimodal', help='the chosen model,should be one of : visual_model, audio_model, word_model, bimodal_model, trimodal_model')
    parser.add_argument('--task', default='all', help="The regression task for arousal, valence or emotion categories.")
    parser.add_argument('--fusion', action='store_true', help="Include it if the models separated by comma should be fusioned.")
    parser.add_argument('--fusion_type', default='all', help="The type of fusion: early, late, or all.")
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs.')
    parser.add_argument('--pretrained', action='store_true', help="Use a pretrained model.")
    args = parser.parse_args()
    tf.compat.v1.flags.DEFINE_string("model", args.model, 'the chosen model,should be one of : visual_model, audio_model, word_model, bimodal_model, trimodal_model')
    tf.compat.v1.flags.DEFINE_integer("epochs", args.epochs, "Number of epochs for training.")
    tf.compat.v1.flags.DEFINE_string("task", args.task, "The regression task for arousal, valence or emotion categories.")
    tf.compat.v1.flags.DEFINE_boolean("fusion", args.fusion, "Include it if the models separated by comma should be fusioned.")
    tf.compat.v1.flags.DEFINE_string("fusion_type", args.fusion_type, "The type of fusion: early, late, or all.")
    tf.compat.v1.flags.DEFINE_boolean("pretrained", args.pretrained, "Use a pretrained model.")
    main()