'''
Defining all the global variables in this cell
'''
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_DEPTH = 3

G_WIN_SIZE= 12
G_DIM = G_WIN_SIZE*G_WIN_SIZE*IMG_DEPTH

STD_VAR = 0.11

LOC_DIM = 2 # the number of dimensions for the locations are just x and y so 2
GLIMPSE_FC1 = 256
GLIMPSE_FC2 = 576

LSTM_HIDDEN = 576

NUM_GLIMPSES = 4


NUM_CLASSES = 10

BASE_OUT = 1
SCALE = 3
PAD_SIZE = G_WIN_SIZE * (2 ** (SCALE-1))

NUM_EPISODES = 20
