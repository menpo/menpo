import numpy as np

def lbp(input_image, radius=1, number_of_samples=8, mapping_type='u2', output_mode='dense'):
    #mapping_type = 'u2' for uniform, 'ri' for rotation-invariant,
    #               'riu2' for uniform rotation-invariant, 0 for no mapping
    #output_mode = 'h' for histogram, 'nh' for normalizedHistogram, 'dense' for dense LBP image

    return radius


def _get_mapping(number_of_samples, mapping_type):
    #returns a mapping table for LBP codes in a neighbourhood of number_of_samples sampling
    #mapping_type = 'u2' for uniform, 'ri' for rotation-invariant, 'riu2' for uniform rotation-invariant

    table = np.arange(0, 2**number_of_samples)
    new_max = 0  # number of patterns in the resulting LBP code
    index = 0

    if mapping_type == 'u2':  # Uniform 2
        new_max = number_of_samples*(number_of_samples-1) + 3
        for i in range(0, 2**number_of_samples):
            j = bitset(bitshift(i,1,number_of_samples),1,bitget(i,number_of_samples)) # rotate left
            numt = sum(bitget(bitxor(i,j),1:number_of_samples)) # number of 1->0 and
            #0->1 transitions
            #in binary string
            #x is equal to the
            #number of 1-bits in
            #XOR(x,Rotate left(x))
            if numt <= 2:
                table[i] = index
                index += 1
            else:
                table[i] = new_max - 1

    if mapping_type == 'ri':  # Rotation invariant
        tmp_map = np.zeros((2**number_of_samples, 1)) - 1
        for i in range(0, 2**number_of_samples):
            rm = i
            r = i
            for j in range(1, number_of_samples):
                r = bitset(bitshift(r,1,number_of_samples),1,bitget(r,number_of_samples))  # rotate left
                if r < rm:
                    rm = r
            if tmp_map[rm] < 0:
                tmp_map[rm] = new_max
                new_max += 1
            table[i] = tmp_map[rm]

    if mapping_type == 'riu2':  # Uniform & Rotation invariant
        new_max = number_of_samples + 2
        for i in range(0, 2**number_of_samples):
            j = bitset(bitshift(i,1,number_of_samples),1,bitget(i,number_of_samples))  # rotate left
            numt = sum(bitget(bitxor(i,j),1:number_of_samples))
            if numt <= 2:
                table[i] = sum(bitget(i,1:number_of_samples))
            else:
                table[i] = number_of_samples + 1

    num = new_max
    return table, number_of_samples, num

