import numpy as np
from utilities.util import normalize


one_data = np.load('data/classifier_data/ONE.npy') 
quant = 1

def hand():
    for data in one_data[:quant]:
        plot_hand(data)
def normalized():
    for data in one_data[:quant]:
        plot_hand(normalize(data)[0])
    plt.show()

def check_scale():
    raw_hand = one_data[0]
    norm_hand = normalize(raw_hand)[0]
    angles = []
    is_not_zero = lambda x: not np.array_equal(x, [0,0,0])    
    for hand in raw_hand, norm_hand:
        vectors = np.array([hand[i[1]]-hand[i[0]] for i in ((0,5),(0,17),(5,17))])
        orient_vector_ang = np.array([[1,1,0],[-1,0,1],[0,-1,-1]])

        pairs = [ori*vectors for ori in np.nditer(orient_vector_ang)]
        pairs = [list(filter(is_not_zero, pair)) for pair in pairs]
        print(pairs)
        angles += [angle_between(pair[0], pair[1]) for pair in pairs]
    print(angles)

check_scale()