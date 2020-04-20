import os, sys, datetime
import numpy as np
import keras
import gc
from copy import deepcopy

# sys.path.append('../dataset_generation/random_configs_poisson/')
from dataset_utils import rotate_about_x

def scale_linear(inp_arr, max_val, min_val):
    raise NotImplementedError

def unscale_linear(inp_arr, max_val, min_val):
    raise NotImplementedError

def scale_z_score(inp_arr, mean, std_dev):
    return (inp_arr - mean) / std_dev

def unscale_z_score(inp_arr, mean, std_dev):
    return (inp_arr * std_dev) + mean

def scale_data(pos=None, frc=None, scaling_dict=None, pos_statistics_dict=None, frc_statistics_dict=None):
    assert pos is not None
    axes_names = ['x', 'y', 'z']

    # Scaling pos
    old_shape = pos.shape
    for i, axis_name in enumerate(axes_names):
        scale_type = scaling_dict['pos_'+axis_name]
        if scale_type is not None:
            if scale_type == 'z_score':
                pos[:, :, i] = scale_z_score(pos[:, :, i], mean=pos_statistics_dict[axis_name+'_mean'], std_dev=pos_statistics_dict[axis_name+'_std_dev'])
            elif scale_type == 'linear':
                raise NotImplementedError
    assert pos.shape == old_shape

    # Scaling frc
    if frc is not None:
        old_shape = frc.shape
        for i, axis_name in enumerate(axes_names):
            scale_type = scaling_dict['frc_'+axis_name]
            if scale_type is not None:
                if scale_type == 'z_score':
                    frc[:, i] = scale_z_score(frc[:, i], mean=frc_statistics_dict[axis_name+'_mean'], std_dev=frc_statistics_dict[axis_name+'_std_dev'])
                elif scale_type == 'linear':
                    raise NotImplementedError
        assert frc.shape == old_shape

    return pos, frc

def unscale_forces(frc, scaling_dict, frc_statistics_dict):
    axes_names = ['x', 'y', 'z']
    old_shape = frc.shape
    for i, axis_name in enumerate(axes_names):
        scale_type = scaling_dict['frc_'+axis_name]
        if scale_type is not None:
            if scale_type == 'z_score':
                frc[:, i] = unscale_z_score(frc[:, i], mean=frc_statistics_dict[axis_name+'_mean'], std_dev=frc_statistics_dict[axis_name+'_std_dev'])
            elif scale_type == 'linear':
                raise NotImplementedError
    assert frc.shape == old_shape
    return frc

class NBatchLogger(keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    ref: https://github.com/keras-team/keras/issues/2850
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}
    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('\nstep: {}/{} ... {} , Time: {} IST'.format(self.step,
                                          self.params['steps'],
                                          metrics_log, datetime.datetime.now().strftime('%H:%M:%S')))
            self.metric_cache.clear()
        if abs(self.step - self.params['steps']) <= 1:
            self.step = 0
            #print(gc.get_objects())
            #print(gc.get_stats())
            #gc.collect()

class DataGenerator(keras.utils.Sequence):
    
    'Generates data for Keras'
    def __init__(self, num_frames, num_atoms, num_rotations, num_nearest_atoms,
                    dataset_folder, list_IDs, pos_arrays, frc_arrays,
                    pos_statistics_dict, frc_statistics_dict, scaling_dict,
                    batch_size, shuffle):
        # mode == train or validation
        'Initialization'
        
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_atoms = num_atoms
        self.num_rotations = num_rotations
        self.num_nearest_atoms = num_nearest_atoms
        self.shuffle = shuffle
        self.dataset_folder = dataset_folder
        
        self.list_IDs = list_IDs
        self.pos_arrays = pos_arrays
        self.frc_arrays = frc_arrays

        self.scaling_dict = scaling_dict
        self.pos_statistics_dict = pos_statistics_dict
        self.frc_statistics_dict = frc_statistics_dict

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #print(index)
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.num_nearest_atoms, 3))
        y = np.empty((self.batch_size, 3))

        # Generate data
        for i, (frame_no, atom_no, rotation_degree) in enumerate(list_IDs_temp):
            # Store sample
            #print(frame_no, atom_no)
            X[i,] = deepcopy(self.pos_arrays[frame_no][atom_no])
            y[i,] = deepcopy(self.frc_arrays[frame_no][atom_no])

            # rotations will happen step-wise, counterclockwise
            X[i,] = [rotate_about_x(X[i,][j], rotation_degree) for j in range(X[i,].shape[0])]
            X[i,] = np.array(X[i,])
            y[i,] = rotate_about_x(y[i,], rotation_degree)

        X, y = scale_data(X, y, self.scaling_dict, self.pos_statistics_dict, self.frc_statistics_dict)
        return np.reshape(X, (-1, self.num_nearest_atoms*3)), y

class DataGenerator_multiple_densities(keras.utils.Sequence):
    
    'Generates data for Keras'
    def __init__(self, 
                 #num_frames,
                 num_atoms, num_rotations, num_nearest_atoms,
                    dataset_folder, list_IDs, pos_arrays, frc_arrays,
                    pos_statistics_dict, frc_statistics_dict, scaling_dict,
                    batch_size, shuffle,

                    files_suffix_list_index_mapping,
                    load_dataset_in_ram):
        # mode == train or validation
        'Initialization'
        
        self.batch_size = batch_size
        #self.num_frames = int(num_frames)
        self.num_atoms = num_atoms
        self.num_rotations = num_rotations
        self.num_nearest_atoms = num_nearest_atoms
        self.shuffle = shuffle
        self.dataset_folder = dataset_folder
        
        self.list_IDs = list_IDs
        self.pos_arrays = pos_arrays
        self.frc_arrays = frc_arrays

        self.scaling_dict = scaling_dict
        self.pos_statistics_dict = pos_statistics_dict
        self.frc_statistics_dict = frc_statistics_dict

        self.files_suffix_list_index_mapping = files_suffix_list_index_mapping
        self.load_dataset_in_ram = load_dataset_in_ram

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #print(index)
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.num_nearest_atoms, 3))
        y = np.empty((self.batch_size, 3))

        # Generate data
        for i, (file_suffix, atom_no, rotation_degree) in enumerate(list_IDs_temp):
            atom_no = int(atom_no)
            rotation_degree = float(rotation_degree)
            # Store sample
            #print(frame_no, atom_no)
            #print(type(self.files_suffix_list_index_mapping[file_suffix]), type(atom_no), type(rotation_degree))
            #print(self.pos_arrays[0][0])

            if self.load_dataset_in_ram:
                X[i,] = deepcopy(self.pos_arrays[self.files_suffix_list_index_mapping[file_suffix]][atom_no])
                y[i,] = deepcopy(self.frc_arrays[self.files_suffix_list_index_mapping[file_suffix]][atom_no])
            else:
                box_size_folder = file_suffix.split('__')[1] # Get folder name from the filename string
                X[i,] = np.load(os.path.join(self.dataset_folder, box_size_folder, 'pos_arr_'+file_suffix))['arr_0'][atom_no]
                y[i,] = np.load(os.path.join(self.dataset_folder, box_size_folder, 'frc_arr_'+file_suffix))['arr_0'][atom_no]

            # rotations will happen step-wise, counterclockwise
            X[i,] = [rotate_about_x(X[i,][j], rotation_degree) for j in range(X[i,].shape[0])]
            X[i,] = np.array(X[i,])
            y[i,] = rotate_about_x(y[i,], rotation_degree)

        X, y = scale_data(X, y, self.scaling_dict, self.pos_statistics_dict, self.frc_statistics_dict)
        return np.reshape(X, (-1, self.num_nearest_atoms*3)), y


if __name__ == '__main__':
    # Only for debugging
    # Parameters
    params = {
                'num_frames': 10,
                'num_atoms': 96,
                'num_rotations': 90,
                'num_nearest_atoms': 50,
                'dataset_folder': '/home/test/random_configs_poisson/dataset_3/final_data/50_nearest_0_rotations/unscaled_numpy_matrices',
                'batch_size': 64,
                'shuffle': True
            }

    # Generators
    training_generator = DataGenerator(mode='train', **params)
    for i in range(len(training_generator)):
        training_generator.__getitem__(i)
    #training_generator.__getitem__(1)
    validation_generator = DataGenerator(mode='validation', **params)
    validation_generator.__getitem__(0)
