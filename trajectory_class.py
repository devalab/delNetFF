import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import os
import multiprocessing
from simtk.openmm import Vec3
from simtk.unit import *
from datetime import datetime
import progressbar, pickle
from copy import deepcopy
from math import *
#import nn_model
from termcolor import colored
import time

import sys
this_file_folder = os.path.dirname(os.path.realpath(__file__))

def timing(f):
    # measure time taken by a function call
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

def get_max_processes(no_of_frames):
    try:
        max_processes = int(os.environ["SLURM_CPUS_ON_NODE"])
    except:
        max_processes = multiprocessing.cpu_count()
    return min(no_of_frames, max_processes)

def interval_divide(min_val, max_val, intervals, print_flag=False):
    assert intervals != 0
    interval_size = (max_val - min_val) / intervals
    result = []
    start = min_val
    end = min_val + interval_size
    while True:
        result.append([int(start), int(end)])
        start = end + 1
        end = end + interval_size
        if len(result) == intervals:
            break
    if result[-1][1] != max_val:
        result[-1][1] = max_val
    if print_flag:
        print("Intervals = ", result)
    return result


class Trajectory:

    def __init__(self, box_size, time_step):
        self.frames = multiprocessing.Manager().list()  # This is a multiprocessing list so that it doesnt get copied
        #  across all child processes
        self.no_of_frames = 0
        self.box_size = box_size
        self.time_step = time_step
        self.pos_frames_deleted = 0
        assert self.box_size is not None
        
        self.vel_frames = multiprocessing.Manager().list()  # This is a multiprocessing list so that it doesnt get copied
        self.vel_no_of_frames = 0
        self.vel_frames_deleted = 0

        self.frc_frames = multiprocessing.Manager().list()  # This is a multiprocessing list so that it doesnt get copied
        self.frc_no_of_frames = 0
        self.frc_frames_deleted = 0
        
        self.box_size_list_for_rdf_plotting = None # In case of npt simulation, min image convention requires box size at each step

    def __len__(self):
        return self.no_of_frames

    def __str__(self):
        for i in range(self.no_of_frames):
            print("Frame ", i)
            for j in self.frames[i]:
                print(j)
            print("--------------------------------------------------------------------")
        return ""

    def append_frame(self, x):
        if self.no_of_frames > 0:
            assert len(self.frames[-1]) == len(x)
        self.frames.append(deepcopy(x))
        self.no_of_frames += 1
        return
    
    def append_vel_frame(self, x):
        if self.vel_no_of_frames > 0:
            assert len(self.vel_frames[-1]) == len(x)
        self.vel_frames.append(deepcopy(x))
        self.vel_no_of_frames += 1
        return
    
    def append_frc_frame(self, x):
        if self.frc_no_of_frames > 0:
            assert len(self.frc_frames[-1]) == len(x)
        self.frc_frames.append(deepcopy(x))
        self.frc_no_of_frames += 1
        return

    def delete_frames(self, start_frame, end_frame):
        # delete frames from index start_frame to end_frame inclusive
        n_frames_to_delete = end_frame - start_frame + 1
        assert n_frames_to_delete < self.no_of_frames # keep atleast 1
        indexes_to_keep = [ind for ind in range(0, self.no_of_frames) if ind < start_frame or ind > end_frame]
        self.frames = [self.frames[i] for i in indexes_to_keep]
        self.no_of_frames -= n_frames_to_delete
        self.pos_frames_deleted += n_frames_to_delete
        return
    
    def delete_vel_frames(self, start_frame, end_frame):
        # delete frames from index start_frame to end_frame inclusive
        n_frames_to_delete = end_frame - start_frame + 1
        assert n_frames_to_delete < self.vel_no_of_frames # keep atleast 1
        indexes_to_keep = [ind for ind in range(0, self.vel_no_of_frames) if ind < start_frame or ind > end_frame]
        self.vel_frames = [self.vel_frames[i] for i in indexes_to_keep]
        self.vel_no_of_frames -= n_frames_to_delete
        self.vel_frames_deleted += n_frames_to_delete
        return

    def delete_frc_frames(self, start_frame, end_frame):
        # delete frames from index start_frame to end_frame inclusive
        n_frames_to_delete = end_frame - start_frame + 1
        assert n_frames_to_delete < self.frc_no_of_frames # keep atleast 1
        indexes_to_keep = [ind for ind in range(0, self.frc_no_of_frames) if ind < start_frame or ind > end_frame]
        self.frc_frames = [self.frc_frames[i] for i in indexes_to_keep]
        self.frc_no_of_frames -= n_frames_to_delete
        self.frc_frames_deleted += n_frames_to_delete
        return

    def dist_between_atoms_min_image_convention(self, atom_1_coors, atom_2_coors):
        # atom_1_coors and atom_2_coors are openmm unit types
        # access value of unit by doing ._value
        # box_vector = [16.7, 16.7, 16.7]

        # convert to np array
        # if isinstance(atom_1_coors[0], float) and isinstance(atom_2_coors[0], float):
        #     arr = np.row_stack((atom_1_coors, atom_2_coors))
        # else:
        #     arr = np.row_stack(([x._value for x in atom_1_coors], [x._value for x in atom_2_coors]))
        box_size_converted = [self.box_size[i].in_units_of(angstroms) for i in range(3)]
        assert atom_1_coors.unit == atom_2_coors.unit and atom_1_coors.unit == box_size_converted[0].unit
        dist = abs(atom_2_coors - atom_1_coors)
        for i in range(3):
            if dist[i] > 0.5 * box_size_converted[i]:
                dist[i] -= (box_size_converted[i] * (round(dist[i] / box_size_converted[i])))
        ret = sqrt(sum([(dist[i]._value)**2 for i in range(3)])) * dist.unit
        return ret

    def wrap(self, x, y, z):
        raise NotImplementedError # Didn't do the mod - floating point thingy 

    def plot_xyz_coors(self, bins=500):
        title_strs = ['x', 'y', 'z']
        for i in range(3):
            coor_data = [ atom_coor[i]._value for frame in self.frames for atom_coor in frame ]
            #print(coor_data)
            #exit(0)
            plt.hist(coor_data, bins=bins)
            plt.title(str("Histogram of " + title_strs[i] + " coordinate"))
            plt.show()


    def save_as_xyz(self, filename, skip_every_n_frames=1):
        print("Trajectory dump to ", filename, " started.")
        bar = progressbar.ProgressBar(max_value=len(self.frames)/skip_every_n_frames)
        
        if skip_every_n_frames > 1:
            skipped_frames = [self.frames[i] for i in range(0, len(self.frames), skip_every_n_frames)]
        skipped_frames = self.frames
        with open(filename, 'w') as out_file:
            for frame_no, frame in enumerate(skipped_frames):
                out_file.write(str(len(frame)) + '\n')
                #out_file.write(" generated by psp" + '\n')
                out_file.write(str('Frame_no = ' + str(frame_no) + ", Time = " + str(self.time_step*frame_no) +
                                   " Distance_unit = angstroms" +
                "\n"))
                for c in frame: # openmm uses nanometre, multiply by 10 to dump angstroms
                    out_file.write(str("   Ar" + "\t" + str(10*c[0]._value) + "\t" + str(10*c[1]._value) + "\t" +
                                       str(10*c[
                                                                                                                    2]._value) + "\n"))
                bar.update(frame_no)
        print()
        

        print("Trajectory dump to ", filename, " complete.")
        return

    def dump_pickle(self, filename):
        with open(filename, 'wb') as out_file:
            pickle.dump(self, out_file)
        return


    def get_wrapped_copy_of_frame(self, frame_no):
        # https://en.wikipedia.org/wiki/Periodic_boundary_conditions
        
        # The np.apply_along_axis call below is supposed to replace the following for loop. Run the for loop if you wanna debug or check.
        #ret = deepcopy(self.frames[frame_no])
        #len_ret = len(ret)
        #for i in range(len_ret):
        #    tmp = []
        #    for j in range(3):
        #        assert self.box_size[j].unit == ret[i][j].unit
        #        x = floor(ret[i][j] / self.box_size[j]) * self.box_size[j]
        #        tmp.append(x._value)
        #    tmp = np.array(tmp)
        #    tmp *= ret.unit
        #    ret[i] -= tmp
        #return ret2

        ret2 = deepcopy(self.frames[frame_no])
        temp_box_size = np.array([self.box_size[i]._value for i in range(3)]) * self.box_size[0].unit
        ret2 = np.apply_along_axis(lambda row_arr: row_arr - [(floor((row_arr[i]*ret2.unit) / temp_box_size[i]) * temp_box_size[i]).in_units_of(ret2.unit)._value for i in range(3)], axis=-1, arr=ret2) * ret2.unit
        
        #assert np.array_equal(ret, ret2)
        return ret2

    def wrap_trajectory(self):
        for i in range(self.frames.shape[0]):
            self.frames[i] = self.get_wrapped_copy_of_frame(i)
        return


    def separate_values_and_unit(self, inp):
        inp_unit = inp[0].unit
        ret = deepcopy(inp)
        ret = [i/inp_unit for i in ret]
        return np.array(ret), inp_unit

    def stack_numpy_unit_array(self, inp):
        ret, inp_unit = self.separate_values_and_unit(inp)        
        ret = np.stack(ret)
        return ret * inp_unit

    def vstack_numpy_unit_array(self, inp):
        ret, inp_unit = self.separate_values_and_unit(inp)        
        ret = np.vstack(ret)
        return ret * inp_unit

    # def convert_numpy_unit_array(self, inp, destination_unit):
    #     ret, inp_unit = self.separate_values_and_unit(inp)        
    #     ret = np.vstack(ret)
    #     x = 1 * inp_unit
    #     x = x._in_
    #     return ret * inp_unit

    #@timing
    def get_env_for_atoms(self, st_atom_no, en_atom_no, wrapped_frame, nearest_no_of_atoms=None):
        ret = []
        #print('get_env_for_atoms ', st_atom_no, '/', en_atom_no)
        for atom_no in range(st_atom_no, en_atom_no+1):
            wrapped_frame_copy = deepcopy(wrapped_frame)
            atom_coors = deepcopy(wrapped_frame_copy[atom_no])
            wrapped_frame_copy -= atom_coors
            wrapped_frame_copy = np.delete(wrapped_frame_copy, atom_no, 0) * wrapped_frame_copy[0][0].unit # remove that atom's entry
            
            #print("Started getting nearest copy", atom_no, '/', en_atom_no)
            # Get nearest image of each atom wrt central atom
            # The following 2 np.where statements are supposed to replicate the following code snippet, but faster
            #for i in range(wrapped_frame_copy.shape[0]):
            #    for j in range(3):
            #        if wrapped_frame_copy[i][j] < -self.box_size[j] * 0.5:
            #            wrapped_frame_copy[i][j] += self.box_size[j]
            #        elif wrapped_frame_copy[i][j] > self.box_size[j] * 0.5:
            #            wrapped_frame_copy[i][j] -= self.box_size[j]
            
            temp_box_size = np.array([self.box_size[i]._value for i in range(3)]) * self.box_size[0].unit
            wrapped_frame_copy = np.where(wrapped_frame_copy < -temp_box_size*0.5, wrapped_frame_copy+temp_box_size, wrapped_frame_copy) * wrapped_frame_copy.unit
            wrapped_frame_copy = np.where(wrapped_frame_copy >  temp_box_size*0.5, wrapped_frame_copy-temp_box_size, wrapped_frame_copy) * wrapped_frame_copy.unit
            #print("Finished getting nearest copy", atom_no, '/', en_atom_no)


            distances_from_zero_zero = np.linalg.norm(wrapped_frame_copy, axis=1)
            order_of_indices = np.argsort(distances_from_zero_zero)
            wrapped_frame_copy = wrapped_frame_copy[order_of_indices]
            if nearest_no_of_atoms is not None:
                wrapped_frame_copy = wrapped_frame_copy[:nearest_no_of_atoms]
            #wrapped_frame_copy = np.asarray([wrapped_frame_copy])
            #print(wrapped_frame_copy.shape)
            ret.append(deepcopy(wrapped_frame_copy))
            # if atom_no == st_atom_no:
            #     ret = np.stack(deepcopy(wrapped_frame_copy))
            # else:
            #     ret = np.vstack((ret, wrapped_frame_copy))
            # print(ret.shape)
            #if st_atom_no == 0:
            #    print(st_atom_no, en_atom_no, ret.shape)
        #print(ret.unit)
            
        ret = self.stack_numpy_unit_array(ret)
        assert ret.shape[0] == en_atom_no-st_atom_no+1
        return ret

    def get_env_for_atoms_helper(self, args):
        return self.get_env_for_atoms(*args)

    #@timing
    def get_envs_for_frame(self, frame_no, nearest_no_of_atoms=None, process_pool=None):
        if process_pool is None:
            raise ValueError("Pass a process pool so that function can do its job")

        frame_no -= self.pos_frames_deleted # correct for deleted_frames
        n_atoms = len(self.frames[frame_no])
        assert nearest_no_of_atoms <= n_atoms-1
        
        wrapped_frame = self.get_wrapped_copy_of_frame(frame_no)
        
        atom_no_ranges = interval_divide(0, n_atoms-1, process_pool._processes)
        get_envs_inputs = [ [i[0], i[1], wrapped_frame, nearest_no_of_atoms] for i in atom_no_ranges ]
        atom_envs = process_pool.map(self.get_env_for_atoms_helper, get_envs_inputs)
        #print([x.shape for x in atom_envs])
        
        atom_envs = self.vstack_numpy_unit_array(atom_envs)
        if nearest_no_of_atoms is None:
            assert atom_envs.shape == (n_atoms, n_atoms-1, 3)
        else:
            assert atom_envs.shape == (n_atoms, nearest_no_of_atoms, 3)
        return atom_envs



