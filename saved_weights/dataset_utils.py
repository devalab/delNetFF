import numpy as np
from copy import deepcopy
import multiprocessing, os
from termcolor import colored
import os

# https://stackoverflow.com/questions/43507491/imprecision-with-rotation-matrix-to-align-a-vector-to-an-axis
def get_rotation_matrix(i_v, unit=None):
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [1.0, 0.0, 0.0] # defaults to x-axis
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, unit)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)

    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0, -w,  v],
            [ w,  0, -u],
            [-v,  u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )

def align_nearest_to_x_axis(pos, forces=None):
    pos_magnitudes = np.linalg.norm(pos, axis=1)

    nearest_vector = deepcopy(pos[0])
    rotation_matrix = get_rotation_matrix(nearest_vector)

    #print("Before roation pos = ", pos)
    pos = np.array([np.matmul(rotation_matrix, i) for i in pos])
    #print("After roation pos = ", pos)
    new_pos_magnitudes = np.linalg.norm(pos, axis=1)
    try:
        assert np.allclose(pos_magnitudes, new_pos_magnitudes) # verify if magnitudes are conserved after rotation
    except:
        print("pos magnitudes = ", pos_magnitudes)
        print("pos magnitudes after rotation = ", new_pos_magnitudes)
        raise AssertionError("Np allclose failed.")

    if forces is not None:
        #print("Before roation frc = ", forces)
        frc_magnitudes = np.linalg.norm(forces, axis=0)
        forces = np.matmul(rotation_matrix, forces)
        new_frc_magnitudes = np.linalg.norm(forces, axis=0)
        #print("After roation frc = ", forces)
        assert np.allclose(frc_magnitudes, new_frc_magnitudes) # verify if magnitudes are conserved after rotation

    return pos, forces

def Rx ( theta ):
    # numpy takes inputs in radians
    
    # Counterclockwise rotation by theta degrees
    # about x-axis
    
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    rx = [
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
         ]
    return np.array(rx)

def Ry ( theta ):
    # numpy takes inputs in radians
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    rx = [
            [c,  0, s],
            [0,  1, 0],
            [-s, 0, c]
         ]
    return np.array(rx)

def Rz ( theta ):
    # numpy takes inputs in radians
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    rx = [
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
         ]
    return np.array(rx)

def rotate_about_x (x, degrees):
    ret = np.matmul(Rx(degrees), x)
    return ret

axes_names = ['x', 'y', 'z']

lock = multiprocessing.Lock()

def get_pos_and_forces_from_file(pos_filename, number_of_rotations, dataset_folder):
    coors = np.load(os.path.join(dataset_folder, os.path.basename(os.path.dirname(pos_filename)), 'pos_arr_'+os.path.basename(pos_filename)+'_unscaled.npz'))['arr_0']
    frcs = np.load(os.path.join(dataset_folder, os.path.basename(os.path.dirname(pos_filename)), 'frc_arr_'+os.path.basename(pos_filename)+'_unscaled.npz'))['arr_0']
    assert coors.shape[0] == frcs.shape[0]
    if number_of_rotations == 0:
        angles_list = [0.0]
    else:
        angles_list = np.arange(0.0, 360.0, 360.0/number_of_rotations)
    coors_rotated = []
    frcs_rotated = []
    for coor, frc in zip(coors, frcs):
        for angle in angles_list:
            coors_rotated.append(np.array([ rotate_about_x(deepcopy(i), angle) for i in coor]))
            frcs_rotated.append(rotate_about_x(deepcopy(frc), angle))
    coors_rotated = np.array(coors_rotated)
    frcs_rotated = np.array(frcs_rotated)
    return coors_rotated, frcs_rotated


def update_means():
    # Update means
    position_statistics['overall_mean'] /= position_statistics['num_total_coordinates']
    assert position_statistics['num_total_coordinates'] % 3 == 0
    for i, axis_name in enumerate(axes_names):
        position_statistics[axis_name+'_mean'] /= (position_statistics['num_total_coordinates'] / 3)

    forces_statistics['overall_mean'] /= forces_statistics['num_total_coordinates']
    assert forces_statistics['num_total_coordinates'] % 3 == 0
    for i, axis_name in enumerate(axes_names):
        forces_statistics[axis_name+'_mean'] /= (forces_statistics['num_total_coordinates'] / 3)
    return

def finalize_statistics_dicts():

    # Update Variances
    position_statistics['overall_variance'] /= position_statistics['num_total_coordinates']
    assert position_statistics['num_total_coordinates'] % 3 == 0
    for i, axis_name in enumerate(axes_names):
        position_statistics[axis_name+'_variance'] /= (position_statistics['num_total_coordinates'] / 3)

    forces_statistics['overall_variance'] /= forces_statistics['num_total_coordinates']
    assert forces_statistics['num_total_coordinates'] % 3 == 0
    for i, axis_name in enumerate(axes_names):
        forces_statistics[axis_name+'_variance'] /= (forces_statistics['num_total_coordinates'] / 3)


    # Update std dev
    position_statistics['overall_std_dev'] = np.sqrt(position_statistics['overall_variance'])
    forces_statistics['overall_std_dev'] = np.sqrt(forces_statistics['overall_variance'])
    for i, axis_name in enumerate(axes_names):
        position_statistics[axis_name+'_std_dev'] = np.sqrt(position_statistics[axis_name+'_variance'])        
        forces_statistics[axis_name+'_std_dev'] = np.sqrt(forces_statistics[axis_name+'_variance'])

    position_statistics['is_dict_complete'] = True
    forces_statistics['is_dict_complete'] = True
    return

def update_variance_for_frame(frame_no, number_of_rotations, dataset_folder):
    print("Updating variance for frame", frame_no)
    coors, frc = get_pos_and_forces_from_file(frame_no, number_of_rotations, dataset_folder)

    pos_overall_variance = np.sum((coors - position_statistics['overall_mean'])**2)
    pos_axes_variances = []
    for i, axis_name in enumerate(axes_names):
        pos_axes_variances.append(np.sum((coors[:, :, i] - position_statistics[axis_name+'_mean'])**2))

    frc_overall_variance = np.sum((frc - forces_statistics['overall_mean'])**2)
    frc_axes_variances = []
    for i, axis_name in enumerate(axes_names):
        frc_axes_variances.append(np.sum((frc[:, i] - forces_statistics[axis_name+'_mean'])**2))

    with lock:
        position_statistics['overall_variance'] += pos_overall_variance
        forces_statistics['overall_variance'] += frc_overall_variance
        for i, axis_name in enumerate(axes_names):
            position_statistics[axis_name + '_variance'] += pos_axes_variances[i]
            forces_statistics[axis_name + '_variance'] += frc_axes_variances[i]
    return

def update_variance_for_frame_helper(args):
    return update_variance_for_frame(*args)

def update_statistics_for_array(frame_pos_array, frame_frc_array):
    
    position_statistics['num_of_frames'] += 1
    forces_statistics['num_of_frames'] += 1
    position_statistics['num_total_coordinates'] += np.size(frame_pos_array)
    forces_statistics['num_total_coordinates'] += np.size(frame_frc_array)

    # Updating counters to calculate mean at the end
    position_statistics['overall_mean'] += np.sum(frame_pos_array)
    for i, axis_name in enumerate(axes_names):
        position_statistics[axis_name+'_mean'] += np.sum(frame_pos_array[:, :, i])
    forces_statistics['overall_mean'] += np.sum(frame_frc_array)
    for i, axis_name in enumerate(axes_names):
        forces_statistics[axis_name+'_mean'] += np.sum(frame_frc_array[:, i])

    # Updating min vals
    position_statistics['overall_min'] = min(position_statistics['overall_min'], np.amin(frame_pos_array))
    for i, axis_name in enumerate(axes_names):
        position_statistics[axis_name+'_min'] = min(position_statistics[axis_name+'_min'], np.amin(frame_pos_array[:, :, i]))
    forces_statistics['overall_min'] = min(forces_statistics['overall_min'], np.amin(frame_frc_array))
    for i, axis_name in enumerate(axes_names):
        forces_statistics[axis_name+'_min'] = min(forces_statistics[axis_name+'_min'], np.amin(frame_frc_array[:, i]))

    # Updating max vals
    position_statistics['overall_max'] = max(position_statistics['overall_max'], np.amax(frame_pos_array))
    for i, axis_name in enumerate(axes_names):
        position_statistics[axis_name+'_max'] = max(position_statistics[axis_name+'_max'], np.amax(frame_pos_array[:, :, i]))
    forces_statistics['overall_max'] = max(forces_statistics['overall_max'], np.amax(frame_frc_array))
    for i, axis_name in enumerate(axes_names):
        forces_statistics[axis_name+'_max'] = max(forces_statistics[axis_name+'_max'], np.amax(frame_frc_array[:, i]))

    # Cant do std dev calculation on the fly. Separate function called at the end.
    return

def update_statistics_for_one_frame(pos_filename, number_of_rotations, dataset_folder):
    coors_rotated, frc_rotated = get_pos_and_forces_from_file(pos_filename, number_of_rotations, dataset_folder)
    with lock:
        print("Updating statistics for frame ", pos_filename)
        update_statistics_for_array(coors_rotated, frc_rotated)
    return

def update_statistics_for_one_frame_helper(args):
    return update_statistics_for_one_frame(*args)

# Dicts to store all important statistical values of a dataset. Globally declared to make it
# available to all processes
position_statistics = multiprocessing.Manager().dict()
forces_statistics = multiprocessing.Manager().dict()

def get_statistics_dict(pos_filenames, number_of_rotations, dataset_folder, n_files_processed):
    
    #initialize_statistics_dicts()
    LARGE_NUMBER = 1000000000000000000000
    LARGE_NEG_NUMBER = -1000000000000000000000
    statistics_dict = {
        'num_of_frames': 0,

        'overall_mean':  0.0,
        'x_mean': 0.0,
        'y_mean': 0.0,
        'z_mean': 0.0,

        'overall_min':  LARGE_NUMBER,
        'overall_max':  LARGE_NEG_NUMBER,
        'x_min': LARGE_NUMBER,
        'x_max': LARGE_NEG_NUMBER,
        'y_min': LARGE_NUMBER,
        'y_max': LARGE_NEG_NUMBER,
        'z_min': LARGE_NUMBER,
        'z_max': LARGE_NEG_NUMBER,

        'overall_std_dev': 0.0,
        'x_std_dev': 0.0,
        'y_std_dev': 0.0,
        'z_std_dev': 0.0,

        'overall_variance': 0.0,
        'x_variance': 0.0,
        'y_variance': 0.0,
        'z_variance': 0.0,

        'num_total_coordinates': 0,
        'number_of_rotations': number_of_rotations,
        'is_dict_complete': False,  # This becomes True only when the dict is fully processed.
    }

    position_statistics.update(statistics_dict)
    position_statistics['description'] = 'Statistics of all position data in the dataset.'

    forces_statistics.update(statistics_dict)
    forces_statistics['description'] = 'Statistics of all force data in the dataset.'
    
    max_processes = multiprocessing.cpu_count()
    process_pool = multiprocessing.Pool(max_processes)


    inps = [[i, number_of_rotations, dataset_folder] for i in pos_filenames]


    process_pool.map(update_statistics_for_one_frame_helper, inps, chunksize=1)
    update_means() # means must be finalized as they are used for variance calculation
    # Divide and get final variance in finalize_statistics_dicts() function
    process_pool.map(update_variance_for_frame_helper, inps, chunksize=1)

    process_pool.close()
    finalize_statistics_dicts()
    # Check whether data from all frames has been considered. This checks if any async
    # overwrite took place.
    #print(position_statistics['num_of_frames'] , n_files_processed)
    assert position_statistics['num_of_frames'] == n_files_processed
    assert forces_statistics['num_of_frames'] == n_files_processed
    assert position_statistics['is_dict_complete'] and forces_statistics['is_dict_complete']

    return dict(position_statistics), dict(forces_statistics)

if __name__ == '__main__':
    pos =  [
                [14.662764658481446,  14.067365054533832,  8.150821418287142],
                [11.594063055425327,  13.37151078442203 ,  6.749916274138395],
                [12.543217980133392,  12.924298003231327,  9.354281707601105],
                [5.986749653825852,   10.309517442594913,  13.355816794263017],
                [8.963156562318694,   11.35469891505295,   3.6511677361868458]
            ]
    pos = np.array(pos)
    #print(pos)
    pos -= pos[0]
    #print('translated = ', pos)
    pos = np.delete(pos, (0), axis=0)
    #print('entry removed = ', pos)
    distances_from_zero_zero = np.linalg.norm(pos, axis=1) # all atom's distances from origin
    order_of_indices = np.argsort(distances_from_zero_zero)
    pos = pos[order_of_indices] # atoms sorted according to distances from origin
    print('sorted = ', pos)
    closest_vec = deepcopy(pos[0])

    frc = [1,1,1]
    pos2, frc2 = align_nearest_to_x_axis(deepcopy(pos), deepcopy(frc))
    #rot_mat *= -1
    #print("rot mat = ", rot_mat)
    rev_rot_mat = get_rotation_matrix([1,0,0], closest_vec/np.linalg.norm(closest_vec))
    print("rev rot mat = ", rev_rot_mat)
    print("Aligned to x = ", pos2)
    pos2 = [np.matmul(rev_rot_mat, i) for i in pos2]
    pos2 = np.array(pos2)
    print("Reversed = ", pos2)
