import numpy as np
import sys, os, time
import argparse
from datetime import datetime
import keras.backend as K
import keras
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Add
from keras import regularizers, optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.callbacks import TensorBoard
#from tf.keras.callbacks import TensorBoard
from keras.initializers import glorot_normal
from sklearn.metrics import r2_score
from termcolor import colored
from pprint import pprint
import pickle, json
from copy import deepcopy
import tensorflow as tf
from random import shuffle

from dataset_handler import DataGenerator_multiple_densities, NBatchLogger, scale_data, unscale_forces

from dataset_utils import *
sys.path.append('../dataset_generation/random_configs_poisson/') # dataset_utils.py file is located here

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def custom_accuracy(percent_tolerance):
     def fn(y_true, y_pred):
         assert percent_tolerance >= 0.0
         true_vals = tf.reshape(y_true, shape=[-1])
         pred_vals = tf.reshape(y_pred, shape=[-1])
         diff = tf.abs(tf.subtract(true_vals, pred_vals))
         true_vals = tf.multiply(true_vals, percent_tolerance/100.0)
         ret = tf.less_equal(diff, tf.abs(true_vals))
         ret = tf.cast(ret, tf.int32)
         #ret = tf.Print(ret, [ret], "Ret Array = ")
         return tf.count_nonzero(ret, dtype=tf.int32) / tf.size(ret)
     return fn


def mse_error(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1) / 2
keras.losses.custom_loss = mse_error

class NeuralNetwork():
    """
    Class for storing all the hyper-parameters related to the network.
    """

    def __init__(self, dummy=False, scaling_dict=None, st_box_size=None, en_box_size=None, manual_statistics_dict_flag=None, load_dataset_in_ram=None):

        # set scaling=False if you do not want to scale positions and forces

        if dummy is True:
            # dont make folders if dummy is true. Set dummy as True only when using it from
            # the simulator, not when training/testing
            return

        assert st_box_size is not None and en_box_size is not None and load_dataset_in_ram is not None

        time_now = datetime.now().strftime('%b-%d-%Y--%H-%M-%S')
        self.model_folder_name = "Model_1__"+ time_now # Folder in which the model will be dumped
        self.log_dir = self.model_folder_name    # Dump logs in the same folder as the model
        self.tensorboard_log_dir = os.path.join(self.log_dir, 'tensorboard_logs')

        self.num_frames = None
        self.st_box_size = st_box_size
        self.en_box_size = en_box_size
        self.manual_statistics_dict_flag = manual_statistics_dict_flag
        self.num_atoms = 96
        self.nearest_atoms = 50
        self.num_rotations = 0
        if self.num_rotations != 0 and self.manual_statistics_dict_flag == True:
            raise ValueError("Stats may be  wrong if num_rotations in dataset prep and num_rotations dont match.")
        self.loss_function = self.mse_error
        keras.losses.custom_loss = self.loss_function

        self.beta = 0. ## Regularization
        self.n_layers = 6
        self.patience = 1 ## How many epochs to wait to half learning rate
        self.n_epochs = 10
        self.train_fraction=0.7

        self.hidden_nodes = [2048, 1024, 512, 256, 128, 64] ## Number of nodes in each layer
        self.keep_prob = [0.7 , 0.7, 0.8, 1, 1, 1]
        self.learning_rate = 5 * 1e-4
        self.batch_size = 2048
        self.filename = '%s_layer__%f_lr__%d_hn__%d_batchsize__%d_epochs__%s'%(self.n_layers , self.learning_rate ,
                                                                                          self.hidden_nodes[0] , self.batch_size,
                                                                                          self.n_epochs, time_now) ## hn = hidden

        self.load_dataset_in_ram = load_dataset_in_ram
        # nodes in first layer

        assert(len(self.hidden_nodes) == self.n_layers)

        ## Create directories and filenames for storing output
        #model_dir = os.path.join('.','Models')
        #log_dir = os.path.join('.','Logs')


        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.model_folder_name):
            os.makedirs(self.model_folder_name)

        if not os.path.exists(self.tensorboard_log_dir):
            os.makedirs(self.tensorboard_log_dir)

        self.model_save = os.path.join(self.model_folder_name, '%s.h5'%(self.filename))
        self.log_save = os.path.join(self.log_dir , '%s.log'%(self.filename))

        self.model = None  # inittialize model with None. NN model will be stored after create_model is called

        self.scaling = all(v is not None for k, v in scaling_dict.items())
        # This sets scaling=False if all values in scaling_dict are None. scaling=False avoids
        # extra computations as stats are not required
        
        self.scaling_dict = scaling_dict
        if self.scaling:
            # these will be set when the set_scale function is called
            if self.scaling_dict is None:
                raise ValueError('Specify scaling details.')
            for k, v in scaling_dict.items():
                if v not in [None, 'linear', 'z_score']:
                    raise NotImplementedError('Unsupported scaling passed.')
        self.pos_statistics_dict = None
        self.frc_statistics_dict = None

    def initialize_generators(self, dataset_folder):
        
        if self.num_rotations == 0:
            angles_list = [0.0]
        else:
            angles_list = np.arange(0.0, 360.0, 360.0/self.num_rotations)

        dataset_folder_contents = os.listdir(dataset_folder)
        dataset_folder_contents = [box_folder_name for box_folder_name in dataset_folder_contents if "box_size" in box_folder_name] # just keep the box_size folders, ignore rest
        dataset_folder_contents = [box_folder_name for box_folder_name in dataset_folder_contents if float(box_folder_name.split('_')[-1]) >= self.st_box_size and float(box_folder_name.split('_')[-1]) <= self.en_box_size] # filter out box_sizes

        print("Making all files suffix list started.")
        all_files_suffix_list = []
        for box_folder_name in dataset_folder_contents:
            dir_name = os.path.join(dataset_folder, box_folder_name)
            print(colored("Using 2500 frames for each box size", 'red'))
            box_folder_file_suffix_list = os.listdir(dir_name)
            box_folder_file_suffix_list = [box_folder_file.replace('pos_arr_', '') for box_folder_file in box_folder_file_suffix_list if 'pos_arr' in box_folder_file] # Store only pos_arr file names
            box_folder_file_suffix_list = box_folder_file_suffix_list[:2500]
            all_files_suffix_list += box_folder_file_suffix_list
            print(len(box_folder_file_suffix_list))
        print("Making all files suffix list ended.")
        
        all_files_suffix_list.sort() # To maintain deterministic list
        files_suffix_list_index_mapping = {}
        for index, suffix in enumerate(all_files_suffix_list):
            files_suffix_list_index_mapping[suffix] = index
        #pprint(files_suffix_list_index_mapping)

        all_files_suffix_and_atom_numbers_and_angles = np.array([(file_suffix, atom_no, angle) for file_suffix in all_files_suffix_list for atom_no in range(self.num_atoms) for angle in angles_list]) # Cross product
        np.random.shuffle(all_files_suffix_and_atom_numbers_and_angles)

        self.num_frames = all_files_suffix_and_atom_numbers_and_angles.shape[0]
        print("Total unrotated  datapoints = ", self.num_frames)
        train_end_frame = int(np.floor(self.num_frames * self.train_fraction))
        train_list_IDs = all_files_suffix_and_atom_numbers_and_angles[:train_end_frame]
        validation_list_IDs = all_files_suffix_and_atom_numbers_and_angles[train_end_frame:]

        if self.scaling:
            if self.manual_statistics_dict_flag:
                pos_statistics_dict_filename = os.path.join(dataset_folder, "position_statistics_dict.npz")
                frc_statistics_dict_filename = os.path.join(dataset_folder, "forces_statistics_dict.npz")
                self.pos_statistics_dict = np.load(pos_statistics_dict_filename)['arr_0'].item()
                self.frc_statistics_dict = np.load(frc_statistics_dict_filename)['arr_0'].item()
            else:
                raise NotImplementedError("For multiple densities this is not implemented")


        files_suffix_list_index_mapping_len = len(files_suffix_list_index_mapping)

        if self.load_dataset_in_ram:
            pos_arrays = np.empty((files_suffix_list_index_mapping_len, self.num_atoms, self.nearest_atoms, 3))
            frc_arrays = np.empty((files_suffix_list_index_mapping_len, self.num_atoms, 3))
        else:
            pos_arrays = None
            frc_arrays = None

        if self.load_dataset_in_ram:
            print('Starting to load all pos and frc arrays into RAM')
            ram_load_progress_counter = 0
            for suffix_name, suffix_index in files_suffix_list_index_mapping.items():
                # Store sample
                box_size_folder = suffix_name.split('__')[1] # Get folder name from the filename string
                pos_arrays[files_suffix_list_index_mapping[suffix_name], ] = np.load(os.path.join(dataset_folder, box_size_folder, 'pos_arr_'+suffix_name))['arr_0']
                frc_arrays[files_suffix_list_index_mapping[suffix_name], ] = np.load(os.path.join(dataset_folder, box_size_folder, 'frc_arr_'+suffix_name))['arr_0']

                ram_load_progress_counter += 1
                print("\t", ram_load_progress_counter, '/', files_suffix_list_index_mapping_len, " done.", end="\r")
            print()
            print('Finished loading all pos and frc arrays into RAM')

        # Parameters
        params = {
                    #'num_frames': self.num_frames,
                    'num_atoms': self.num_atoms,
                    'num_rotations': self.num_rotations,
                    'num_nearest_atoms': self.nearest_atoms,
                    'dataset_folder': dataset_folder,
                    'batch_size': self.batch_size,

                    #'pos_arrays': pos_arrays,
                    #'frc_arrays': frc_arrays,

                    'pos_statistics_dict': self.pos_statistics_dict,
                    'frc_statistics_dict': self.frc_statistics_dict,
                    'scaling_dict': self.scaling_dict,

                    'files_suffix_list_index_mapping': files_suffix_list_index_mapping,
                    'load_dataset_in_ram': self.load_dataset_in_ram
                }

        training_generator = DataGenerator_multiple_densities(list_IDs=train_list_IDs, **params, pos_arrays=pos_arrays, frc_arrays=frc_arrays, shuffle=True)
        validation_generator = DataGenerator_multiple_densities(list_IDs=validation_list_IDs, **params, pos_arrays=pos_arrays, frc_arrays=frc_arrays, shuffle=True)


        return training_generator, validation_generator

    def mse_error(self, y_true, y_pred):
        #import tensorflow as tf
        #y_true = tf.Print(y_true, [y_true])
        return K.mean(K.square(y_true - y_pred), axis=-1) / 2

    def network_layer(self, X, num_nodes, name, keep_prob, beta=None, repeat_val=1, batch_normalize=True):
        """
        Build a fully connected layer and return the output.
        Batch normalization if applicable before non-linearity
        """

        beta = beta if beta is not None else self.beta  # This is because I can't give beta=self.beta as default
        # argument. https://stackoverflow.com/questions/8131942/how-to-pass-a-default-argument-value-of-an-instance-member-to-a-method/8131960

        if batch_normalize:
            layer_output = Dense(units=num_nodes, activation=None, kernel_initializer=glorot_normal(), kernel_regularizer=regularizers.l2(beta), use_bias=False , name=name)(X)
            layer_output = BatchNormalization(name=name+'/BN')(layer_output)
            layer_output = Activation('relu',name=name+'/NL')(layer_output)
        else:
            layer_output = Dense(units=num_nodes, activation='relu',kernel_initializer=glorot_normal(), kernel_regularizer=regularizers.l2(beta), name=name)(X)

        ## Apply non-linearity
        layer_output = Dropout(1-keep_prob)(layer_output)

        return layer_output


    def create_model(self, data_shape, label_shape):
        name = ['_'.join(['L',str(layern)]) for layern in range(1,self.n_layers+1)] ## Name the layers

        input_nodes = data_shape
        hidden_nodes = self.hidden_nodes
        inputs = Input(shape=(input_nodes,))
        output_nodes = label_shape

        output_hidden_1 = self.network_layer(inputs, hidden_nodes[0] , name[0], keep_prob = self.keep_prob[0])
        output_hidden_2 = self.network_layer(output_hidden_1, hidden_nodes[1], name[1], keep_prob = self.keep_prob[1])
        output_hidden_3 = self.network_layer(output_hidden_2, hidden_nodes[2], name[2], keep_prob = self.keep_prob[2])

        output_hidden_4 = self.network_layer(output_hidden_3, hidden_nodes[3], name[3], keep_prob = self.keep_prob[3])
        output_hidden_5 = self.network_layer(output_hidden_4, hidden_nodes[4], name[4], keep_prob = self.keep_prob[4])

        output_hidden_6 = Dense(hidden_nodes[5] , activation=None, kernel_regularizer=regularizers.l2(self.beta), use_bias=False , name='L6-1')(output_hidden_5)
        output_hidden_6 = BatchNormalization(name=str(name[5])+'/BN')(output_hidden_6)

        skip_16_layer = Dense(hidden_nodes[5] , activation=None, name='skip-16', kernel_regularizer=regularizers.l2(self.beta))(output_hidden_1)
        skip_16_layer = BatchNormalization(name='skip-16'+'/BN-1')(skip_16_layer)
        # output_hidden_6 = Add(name='add_1-6')([output_hidden_6 , skip_16_layer])
        # output_hidden_6 = BatchNormalization()(output_hidden_6)
        output_hidden_6 = Activation('relu')(output_hidden_6)

        # No activation/Linear activation applied to final layer.
        final_output = Dense(output_nodes , activation=None)(output_hidden_6)
        model = Model(inputs , final_output)

        print (model.summary())
        return model


    def train(self, dataset_folder):

        """
        Build and train the model.
        Save the model and training log.
        """

        if self.model is None:
            self.model = self.create_model(data_shape=self.nearest_atoms*3, label_shape=3)
            if 'gnode' in os.uname()[1]:
                self.model = keras.utils.multi_gpu_model(self.model) # For parallel gpu training
                print("parallel model summary = ", self.model.summary())

        adam_opt = optimizers.Adam(lr=self.learning_rate)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      patience=self.patience,
                                      factor=0.5,
                                      min_lr=0.5e-10,
                                      verbose=1)

        chk_pt = ModelCheckpoint(self.model_save , monitor='val_loss' , save_best_only=True)

        self.model.compile(optimizer=adam_opt , loss = self.loss_function,
                            metrics=[
                            			'mse',
                            			coeff_determination,
                                        custom_accuracy(10),
                            			custom_accuracy(25),
                                        custom_accuracy(50),
                                        custom_accuracy(100),
                            		])



        tensorboard = TensorBoard(
                                    log_dir=self.tensorboard_log_dir,
                                    write_graph=True,
                                    write_images=True,
                                    #histogram_freq=1,
                                    #write_grad=True,
                                    #update_freq='batch'
                                  )

        out_batch = NBatchLogger(display=1000) # show every x batches

        training_generator, validation_generator = self.initialize_generators(dataset_folder)

        generator_workers = 50
        generator_max_queue_size = 200

        for epoch_number in range(self.n_epochs):
            print(colored("Epoch number: " + str(epoch_number), "red"))
            self.model.fit_generator(
                    generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=generator_workers,
                    epochs=1,
                    callbacks=[reduce_lr , chk_pt, CSVLogger(self.log_save, append=True), out_batch, tensorboard],
                    verbose=1, # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                    #pickle_safe=False
                    max_queue_size=generator_max_queue_size
                    #shuffle=True
                    )
            print("")
            evaluate_generator_ret = self.model.evaluate_generator(
                    generator=training_generator,
                    use_multiprocessing=True,
                    workers=generator_workers,
                    #callbacks=[reduce_lr , chk_pt, CSVLogger(self.log_save, append=False), out_batch, tensorboard],
                    verbose=1, # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                    #pickle_safe=False
                    max_queue_size=generator_max_queue_size
                    #shuffle=True
                )
            print("Evaluate generator on train data output = ")
            print(list(zip(self.model.metrics_names, evaluate_generator_ret)))
            evaluation_metric_log_filename = self.model_folder_name + "_evaluation_metrics.txt"
            with open(evaluation_metric_log_filename, 'a') as evaluation_metric_log_file:
                evaluation_metric_log_file.write(str(list(zip(self.model.metrics_names, evaluate_generator_ret))) + '\n')

    def predict_forces(self, envs_matrix, process_pool):
        """
        Same as evaluate, but without labels. Just return whatever prediction the model has made.
        """
        #print("predict forces start = ", str(datetime.now()))
        envs_matrix_tmp = []
        reverse_rotation_matrices = []
        #print("Before rot envs = ", envs_matrix[0][0:5])
        for i in envs_matrix:
            closest_vec = deepcopy(i[0])
            #print("closest_vec =", closest_vec, " magnitude = ", np.linalg.norm(closest_vec))
            reverse_rotation_matrices.append(get_rotation_matrix([1,0,0], closest_vec/np.linalg.norm(closest_vec)))
            aligned_pos, _ = align_nearest_to_x_axis(i)
            envs_matrix_tmp.append(aligned_pos)
        envs_matrix = np.asarray(envs_matrix_tmp)
        #print("after rot = ", envs_matrix[0][0:5])
        reverse_rotation_matrices = np.asarray(reverse_rotation_matrices)
        #print("After rev rot envs = ", np.array([np.matmul(reverse_rotation_matrices[0], envs_matrix[0][i]) for i in range(envs_matrix[0].shape[0])])[0:5])
        #exit()
        if self.scaling:
            envs_matrix, _ = scale_data(pos=envs_matrix,
                                     frc=None,
                                     scaling_dict=self.scaling_dict,
                                     pos_statistics_dict=self.pos_statistics_dict,
                                     frc_statistics_dict=None)
        envs_matrix = np.reshape(envs_matrix, (-1, self.nearest_atoms*3))
        predicted_forces = self.model.predict(envs_matrix)
        if self.scaling:
            predicted_forces = unscale_forces(frc=predicted_forces,
                                                   scaling_dict=self.scaling_dict,
                                                   frc_statistics_dict=self.frc_statistics_dict)
        predicted_forces = [np.matmul(rev_rot_mat, frc) for rev_rot_mat, frc in zip(reverse_rotation_matrices, predicted_forces)]
        predicted_forces = np.array(predicted_forces)
        #print("predict forces end = ", str(datetime.now()))
        return predicted_forces

    def dump_model(self):

        model_json = self.model.to_json(indent=4)
        json_filename = os.path.join(self.model_folder_name, "model.json")
        with open(json_filename, "w") as json_file:
           json_file.write(model_json)

        hyperparams_filename = os.path.join(self.model_folder_name, "hyperparams.pkl")
        with open(hyperparams_filename, 'wb') as f:
            pickle.dump(self, f)
        return

    def load_all(self, folder_name):
        with open(os.path.join(folder_name, "hyperparams.pkl"), 'rb') as f:
            self = pickle.load(f)

        with open(os.path.join(folder_name, "model.json"), 'r') as f:
            self.model = model_from_json(f.read())

        h5_filename = [f for f in os.listdir(folder_name) if f.endswith('.h5')][0]
        self.model.load_weights(os.path.join(folder_name, h5_filename))

        return self


    def __getstate__(self):
        return dict((k, v) for (k, v) in self.__dict__.items() if k != 'model')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the neural Network Model for force predictions")
    parser.add_argument('-d' , '--datafolder' , help='folder containing the full dataset')
    parser.add_argument('-st_box_size' , '--st_box_size' , help='min box size to consider')
    parser.add_argument('-en_box_size' , '--en_box_size' , help='min box size to consider')
    parser.add_argument('-manual_statistics_dict' , '--manual_statistics_dict', type=str2bool, help='True if supplying statistics dict file manually. If false, model will compute a statistics dict itself.')
    parser.add_argument('-load_dataset_in_ram' , '--load_dataset_in_ram', type=str2bool, help='True if you want to load who;e dataset in RAM. Speeds up training. Set it to false when data cannot fit in ram and needs to be read from the HDD on demand.')


    args = parser.parse_args()
    folder_name = args.datafolder
    st_box_size = float(args.st_box_size)
    en_box_size = float(args.en_box_size)

    manual_statistics_dict_flag = args.manual_statistics_dict
    # if manual_statistics_dict_flag.lower() == 'true':
    #     manual_statistics_dict_flag = True
    # elif manual_statistics_dict_flag.lower() == 'false':
    #     manual_statistics_dict_flag = False
    # else:
    #     raise ValueError()

    load_dataset_in_ram = args.load_dataset_in_ram
    # if load_dataset_in_ram.lower() == 'true':
    #     load_dataset_in_ram = True
    # elif load_dataset_in_ram.lower() == 'false':
    #     load_dataset_in_ram = False
    # else:
    #     raise ValueError()

    scaling_dict = {
        # Possible scaling values are [None, 'linear', 'z_score']
        #Position scaling
        'pos_x': 'z_score',
        'pos_y': 'z_score',
        'pos_z': 'z_score',
        
        #Force scaling
        'frc_x': 'z_score',
        'frc_y': 'z_score',
        'frc_z': 'z_score',
    }
    model_1 = NeuralNetwork(scaling_dict=scaling_dict, st_box_size=st_box_size, en_box_size=en_box_size, manual_statistics_dict_flag=manual_statistics_dict_flag, load_dataset_in_ram=load_dataset_in_ram)

    print('Starting Training ... ')


    #pprint(model_1.__dict__)
    model_1.train(folder_name)
    pprint(model_1.__dict__)
    # model_1.evaluate('./dataset/val.npz')
    model_1.dump_model()
    print(colored('Training and saving complete.', 'green'))

