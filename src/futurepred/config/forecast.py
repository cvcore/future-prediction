from .config_node import MyCfgNode as CN

_C = CN()

########## CONFIGURATION FILE FOR FORECAST MODEL ##########
#                                                         #
#  This file serves as a shared, basic configuration for  #
#  all experiments. Each individual configuation can be   #
#  found in the config/forecast folder.                   #
#                                                         #
###########################################################

########## DATASET ##########

_C.DATASET = CN()

_C.DATASET.PATH = "../../dataset/cityscapes"
_C.DATASET.TYPE = "cityscapes-seq-eps"
_C.DATASET.TARGETS = ['semantic-eps']
_C.DATASET.IMAGE_RESIZE_RATIO = 1.
_C.DATASET.IMAGE_PAD_SIZE = [1025, 2049]
_C.DATASET.TARGET_RESIZE_RATIO = 0.25
_C.DATASET.TARGET_PAD_SIZE = [257, 513]
_C.DATASET.IMAGE_NORM_MEAN = [0.485, 0.456, 0.406]
_C.DATASET.IMAGE_NORM_STD = [0.229, 0.224, 0.225]
_C.DATASET.CROP_SIZE = None
_C.DATASET.RAND_SCALE_TRAIN = [1, 1]
_C.DATASET.RAND_SCALE_VAL = [1, 1]
_C.DATASET.HFLIP_PROB_TRAIN = 0.5
_C.DATASET.HFLIP_PROB_VAL = 0.
_C.DATASET.USE_LOG_DEPTH = True

########## OPTIMIZER ##########

_C.OPTIMIZER = CN()

_C.OPTIMIZER.LR = 0.0003
_C.OPTIMIZER.MAX_EPOCH = 150
_C.OPTIMIZER.LR_SCHEDULER = None

########## GRADIENT SCALER ###########

_C.GRAD_SCALER = CN()

_C.GRAD_SCALER.ENABLE = True
_C.GRAD_SCALER.INIT_SCALE = 10000. # be careful about this value,
                                  # as too large ones can cause
                                  # NaN in the first few epoches.

######### MODEL ###########

_C.MODEL = CN()

_C.MODEL.BS = 2
_C.MODEL.DISTRIBUTED = False
_C.MODEL.LOAD = None
_C.MODEL.STRICT_LOAD = True
_C.MODEL.CHECKPOINT = "checkpoints/checkpoint.pth"
_C.MODEL.CHECKPOINT_INTERVAL = 1800
_C.MODEL.MODEL_DIR = "saved_models"
_C.MODEL.EVAL = False

########## RUNTIME ##########

_C.RUNTIME = CN()

_C.RUNTIME.DEVICE_ID = 0
_C.RUNTIME.MANUAL_SEED = 0
_C.RUNTIME.VERBOSE = False
_C.RUNTIME.VERBOSE_IOU = True
_C.RUNTIME.WORKERS = 4
_C.RUNTIME.USE_AUTOCAST = True
_C.RUNTIME.WATCH_BLENDER_WEIGHTS = False # if set True and MODEL.BLENDER.ENABLE = True, will enable
                                         # logging blender weights to wandb
_C.RUNTIME.SHOW_INDEX_TRAIN = [1]
_C.RUNTIME.SHOW_INDEX_VAL = [1]
_C.RUNTIME.DETECT_ANOMALY = False        # set to True only for debug

########## PREDICTOR ##########

_C.PREDICTOR = CN()

# Prediction general
_C.PREDICTOR.N_FRAMES_HISTORY = 5
_C.PREDICTOR.N_FRAMES_FUTURE = 10
_C.PREDICTOR.N_SAMPLE_EVAL = 5
_C.PREDICTOR.N_SEMANTIC_CLASS = 19
_C.PREDICTOR.DETERMINISTIC = True
_C.PREDICTOR.PREDICT_CURRENT = False
_C.PREDICTOR.OUT_RESIZE = [257, 513]

# Flow
_C.PREDICTOR.FLOW = CN()
_C.PREDICTOR.FLOW.ENABLE = False
_C.PREDICTOR.FLOW.FEATURE_LEVEL = 5 # matches encoder OS=1/32
_C.PREDICTOR.FLOW.FREEZE = True

# Backbone
_C.PREDICTOR.BACKBONE = CN()
_C.PREDICTOR.BACKBONE.TYPE = "panoptic-deeplab"
_C.PREDICTOR.BACKBONE.PATH = "pretrained_models/perception/panoptic_deeplab_R50_os32_cityscapes.pth"
_C.PREDICTOR.BACKBONE.CFG = "pretrained_models/perception/panoptic_deeplab_R50_os32_cityscapes.yaml"
_C.PREDICTOR.BACKBONE.FREEZE = True

# Dynamics
_C.PREDICTOR.DYN = CN()
_C.PREDICTOR.DYN.TYPE = "TempBlock" # options: 'TempBlock' and 'CorrFusion'
_C.PREDICTOR.DYN.IN_CHANNELS = 2048
_C.PREDICTOR.DYN.N_FRAMES = 5 # should equal to N_FRAMES_HISTORY
# Dynamics - TempBlock only
_C.PREDICTOR.DYN.TB_K_SPATIAL = 3
_C.PREDICTOR.DYN.TB_LAYER_CHANNELS = [80, 88, 96, 104]
_C.PREDICTOR.DYN.TB_DROP_LAST = True
# Dynamics - CorrFusion only
_C.PREDICTOR.DYN.N_FUSION = 128
_C.PREDICTOR.DYN.OUT_CHANNELS = 104
# Dynamics - TempShift only
_C.PREDICTOR.DYN.TS_N_DIV = 4 # shift 1/4
_C.PREDICTOR.DYN.TS_LAYER_CHANNELS = [80, 88, 96, 104]
_C.PREDICTOR.DYN.TS_USE_NON_LOCAL = [False, False, False, False]
# Dynamics - ConvLSTM only
_C.PREDICTOR.DYN.CL_LAYER_CHANNELS = [80, 88, 96, 104]
_C.PREDICTOR.DYN.CL_K_SPATIAL = 3
_C.PREDICTOR.DYN.CL_IN_SHAPE = [33, 65] # it should equal to
                                        # input image shape / output strice of perception encoder
# Dynamics - Conv3D only
_C.PREDICTOR.DYN.C3D_K_SPATIAL = 3
_C.PREDICTOR.DYN.C3D_LAYER_CHANNELS = [80, 88, 96, 104]
# Dynamics - ResConv3D only
_C.PREDICTOR.DYN.RC3D_K_SPATIAL = 3
_C.PREDICTOR.DYN.RC3D_LAYER_CHANNELS = [80, 88, 96, 104]
# Dynamics - SepInception only
_C.PREDICTOR.DYN.S3D_LAYER_CHANNELS = [80, 88, 96, 104]
# Dynamics - ConvGRU only
_C.PREDICTOR.DYN.CG_K_SPATIAL = 3
_C.PREDICTOR.DYN.CG_LAYER_CHANNELS = [80, 88, 96, 104]

# Conditioner
_C.PREDICTOR.CONDITIONER = CN()
_C.PREDICTOR.CONDITIONER.DIM_DIST = 16
_C.PREDICTOR.CONDITIONER.DIM_LATENT = 104
_C.PREDICTOR.CONDITIONER.N_LATENT_PAST = 1
_C.PREDICTOR.CONDITIONER.N_LATENT_FUTURE = 3 # 1 (past) + 2 (future)

# Generator
_C.PREDICTOR.GENERATOR = CN()
_C.PREDICTOR.GENERATOR.DIM_DYN = 104
_C.PREDICTOR.GENERATOR.DIM_NOISE = 16 # equal to DIM_DIST from conditioner
_C.PREDICTOR.GENERATOR.DIM_LAYER_BASE = 8
_C.PREDICTOR.GENERATOR.DIM_LAYER = [13, 26, 26, 26, 13] # DIM_OUT = last layer dim * dim_base = 104, equal to dyn feature
_C.PREDICTOR.GENERATOR.N_FUTURE = 10 # should equal to no. of future frames

# Blender
_C.PREDICTOR.BLENDER = CN()
_C.PREDICTOR.BLENDER.ENABLE = False
_C.PREDICTOR.BLENDER.IN_DIM = 104
_C.PREDICTOR.BLENDER.CTX_DIM = 2048
_C.PREDICTOR.BLENDER.OUT_DIM = 104
_C.PREDICTOR.BLENDER.N_FRAME = 5
_C.PREDICTOR.BLENDER.N_LAYER = 1
_C.PREDICTOR.BLENDER.WARP_DIR = 'backward'
_C.PREDICTOR.BLENDER.OUT_INTERMEDIATE = False

# Decoder
_C.PREDICTOR.DECODER = CN()
_C.PREDICTOR.DECODER.TYPE = 'up-conv'
_C.PREDICTOR.DECODER.IN_DIM = 104
_C.PREDICTOR.DECODER.IN_KEY = 'res5'
_C.PREDICTOR.DECODER.OUT_DIM_SEMANTIC = 19
_C.PREDICTOR.DECODER.ODD_UPSAMPLING = True
# Decoder - EfficientPS FPN only
_C.PREDICTOR.DECODER.FPN_CHANNELS = 64
_C.PREDICTOR.DECODER.FPN_HIDDEN_CHANNELS = 32
_C.PREDICTOR.DECODER.FPN_CONFIG_PATH = 'pretrained_models/perception/efficientps/perception_run80_4_city_full.ini'

# Loss
_C.PREDICTOR.WEIGHT_LOSS_PROB = 0.0005
_C.PREDICTOR.WEIGHT_LOSS_PRED = 1.0
_C.PREDICTOR.WEIGHT_LOSS_F2M_FLOW = 0.25
_C.PREDICTOR.WEIGHT_LOSS_GAN_G = 0.1

# Discriminator
_C.DISCRIMINATOR = CN()
_C.DISCRIMINATOR.ENABLE = False
_C.DISCRIMINATOR.TYPE = 'semantic' # valid choices: ['semantic', 'semantic_video']
_C.DISCRIMINATOR.N_CLASSES = 19
_C.DISCRIMINATOR.N_LAYERS = 4
_C.DISCRIMINATOR.N_FEATURES = 64

########## GENERATOR ##########

_C.GENERATOR = CN()

_C.GENERATOR.NORM_SEP_TIME_DIM = False

########## CRITERION ##########

_C.CRITERION = CN()

_C.CRITERION.DISCOUNT_FACTOR = 0.7
_C.CRITERION.MAX_GTFLOW_CONSY_ERR = 0.0003 # max. flow consistency error (for pseudo-groundtruth)
