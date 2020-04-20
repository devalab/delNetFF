def pbc_adjust(double[:,:] coords, double[:] box):
    for i in range(coords.shape[0]):
        for j in range(3):
            coords[i, j] -= round(coords[i, j] / box[j]) * box[j]
