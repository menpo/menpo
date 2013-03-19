import numpy as np
from ibugMM.alignment.nonrigid import TPS


# landmarks used in Principal Warps paper
src_landmarks = np.array([[3.6929, 10.3819],
                          [6.5827,  8.8386],
                          [6.7756, 12.0866],
                          [4.8189, 11.2047],
                          [5.6969, 10.0748]])

tgt_landmarks = np.array([[3.9724, 6.5354],
                          [6.6969, 4.1181],
                          [6.5394, 7.2362],
                          [5.4016, 6.4528],
                          [5.7756, 5.1142]])

## tgt_landmarks = [  6.5354   3.9724; ...
##                    4.1181   6.6969; ...
##                    7.2362   6.5394; ...
##                    6.4528   5.4016; ...
##                    5.1142   5.7756];
#
#tps = TPS(src_landmarks,tgt_landmarks);
#
#
#
#clear src_landmarks;
#clear tgt_landmarks;
#
#src_landmarks = [  0   1; ...
#                 -1   0; ...
#                  0  -1; ...
#                  1   0]; ...
#
#
#tgt_landmarks = [  0   0.75; ...
#                  -1   0.25; ...
#                   0  -1.25; ...
#                   1   0.25]; ...
#
tps = TPS([src_landmarks, src_landmarks], target=tgt_landmarks);


### CALCULATE TPS MAPPING FUNCTION
#
## x = [0:0.1:10]';
## y = [0:0.1:10]';
## affine_free_tps = zeros(size(x,1),size(y,1));
## tpsResult = zeros(size(x,1),size(y,1));
#
## for i = 1:size(x,1)
##     for j = 1:size(y,1)
##     [tempf tempfnoaf] = tps.calculateMapping([x(i) y(j)]);
##     tpsResult(i,j) = tempfnoaf(2);
##     end
## end
###
#res= np.zeros_like(src_landmarks)
#for i in range(src_landmarks.shape[0]):
#    (tempf, tempfnoaf) = tps.calculateMapping(src_landmarks]))
#    res[i] = tempf
tps.build_TPS_matrices()
(tempf, tempfnoaf) = tps.calculateMapping(src_landmarks[:-1])

###
#surf(x,y,tpsResult);
#[V D] = eig(tps.b_energy)
#
#energy = tps.V*tps.b_energy*(tps.V');
