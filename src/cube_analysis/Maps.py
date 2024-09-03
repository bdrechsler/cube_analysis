import numpy as np
import pickle

class Maps:
    def __init__(self, line, line_map, cont_map, wcs, line_thresh=1.5):
        self.line = line
        self.line_map = line_map
        self.cont_map = cont_map
        self.wcs=wcs

        map_shape = np.shape(line_map)
        ratio_map = np.zeros(map_shape)
        
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):

                if line_map[i, j] == 0 or cont_map[i, j] == 0:
                    ratio_map[i, j] = 0
                elif np.isnan(line_map[i, j]) or np.isnan(cont_map[i, j]):
                    ratio_map[i, j] = np.nan
                else:
                    ratio_map[i, j] = line_map[i, j] / cont_map[i, j]

        line_std = np.nanstd(self.line_map)
        no_line_inds = np.where(line_map < line_thresh * line_std)
        ratio_map[no_line_inds] = np.nan
        self.ratio_map = ratio_map

    
    @classmethod
    def load(cls, fname):
        with open(fname, "rb") as f:
            return pickle.load(f)
        
    def write(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)
