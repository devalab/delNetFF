import numpy as np
import mdtraj as md
import sys

def convert_xyz_to_npz(xyz_filename, pdb_filename):
    # pdb filename will be used to build topology for mdtraj
    output_filename = xyz_filename + '.npz'
    print("Reading xyz", xyz_filename)
    traj = md.load_xyz(xyz_filename, top=pdb_filename)
    traj._xyz = traj._xyz.astype(np.double)#[:1000]
    print("Multiplying 10")
    traj._xyz *= 10. # converto to angstroms as mdtraj stores in nanometres
    print("Saving file to ", output_filename)
    np.savez(output_filename, traj=traj._xyz)

    print("Checking....")
    n_atoms = traj._xyz.shape[1]
    #print(traj._xyz[0][0])
    traj._xyz = None # dont use extra memory
    traj._xyz = np.load(output_filename)['traj']
    with open(xyz_filename) as f:
        l = [next(f) for x in range(n_atoms+2)] # check only 1st frame
    l = [i.strip() for i in l]
    assert int(l[0]) == n_atoms
    for i in range(2, n_atoms+2):
        l2 = l[i].split()
        #print(l2)
        for j in range(3):
            #print(traj._xyz[0][i-2][j], float(l2[j+1]))
            assert np.isclose(traj._xyz[0][i-2][j], float(l2[j+1]), atol=0) # atol=0 coz numbers can be much smaller than 1
        
    return

if __name__ == "__main__":

    xyz_filename = sys.argv[1]
    pdb_filename = sys.argv[2]

    convert_xyz_to_npz(xyz_filename, pdb_filename)

