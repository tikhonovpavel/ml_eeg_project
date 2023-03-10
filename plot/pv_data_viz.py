import numpy as np
import pyvista as pv


def pv_viz(eeg_file, dip_file, head_file):
    grid = pv.UnstructuredGrid(head_file)

    eeg = np.load(eeg_file)
    dipole = np.load(dip_file)

    # eeg - red colors
    eeg_data = eeg[:, -1]
    eeg_data = eeg_data[:, np.newaxis]
    z = np.zeros(eeg_data.shape)
    eeg_data = 255 * np.concatenate((eeg_data, z, z), axis=1)

    # blue colours
    dipole_data = dipole[:, -1]
    dipole_data = dipole_data[:, np.newaxis]
    z = np.zeros(dipole_data.shape)
    dipole_data = 255 * np.concatenate((z, z, dipole_data), axis=1)

    # render
    eeg_mesh = pv.PolyData(eeg[:, :-1])
    eeg_mesh["colors"] = eeg_data.astype(np.uint8)

    dipole_mesh = pv.PolyData(dipole[:, :-1])
    dipole_mesh["colors"] = dipole_data.astype(np.uint8)

    pl = pv.Plotter()
    pl.add_mesh(grid, opacity=0.25, color='w')
    pl.add_mesh(eeg_mesh, scalars="colors", rgb=True)
    pl.add_mesh(dipole_mesh, scalars="colors", rgb=True)

    return pl
