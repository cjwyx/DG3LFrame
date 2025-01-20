import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner

from .arch import MegaCRN
from .loss import megacrn_loss

CFG = EasyDict()

from .arch.DGGGL_arch import DGGGL
from .loss.loss import megacrn_loss

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "DGGGL model configuration"
CFG.RUNNER  = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "METR-LA"
CFG.DATASET_TYPE = "Traffic speed"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "DGGGLFrame"
CFG.MODEL.ARCH = DGGGL
CFG.MODEL.PARAM = {
    "num_nodes": 883,
    "input_dim": 1,
    "output_dim": 1,
    "horizon": 12,
    "rnn_units": 32,
    "num_layers":1,
    "cheb_k":3,
    "ycov_dim":1,
    "mem_num":20,
    "mem_dim":80,
    "cl_decay_steps":2000,
    "use_curriculum_learning":True,
    "input_embedding_dim":16,
    "tod_embedding_dim":8,
    "dow_embedding_dim":0,
    "spatial_embedding_dim":0,
    "adaptive_embedding_dim":8,
    "feed_forward_dim": 156,
    "num_layers_t": 1,              
    "num_layers_s": 1,
    "num_at_heads": 4,
    # "dim_formerout": 24,
    "adaptive_embedding_dim_for_gate":12
}
CFG.MODEL.SETUP_GRAPH = True
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = megacrn_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"






# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5
}
CFG.TRAIN.NUM_EPOCHS = 180
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 12
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 4
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 12
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 4
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 12
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 4
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
