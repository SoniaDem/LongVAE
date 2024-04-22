from VAE.utils import get_args
from os.path import join


def default_true(dic, param):
    return False if param in dic and dic[param].lower() == "false" else True


def default_false(dic, param):
    return True if param in dic and dic[param].lower() == "true" else False


def default_float(dic, param, val):
    return float(dic[param]) if param in dic else val


def default_int(dic, param, val):
    return int(dic[param] if param in dic else val)


def get_params(path):

    params = get_args(path)

    param_dict = {"NAME": params["NAME"],
                  "PROJECT_DIR": params["PROJECT_DIR"],
                  "IMAGE_DIR": params["IMAGE_DIR"],
                  "SUBJ_DIR": params["SUBJ_DIR"],
                  "SUBJ_PATH": 'subject_id_key.csv' if "SUBJ_PATH" not in params else params["SUBJ_PATH"],
                  "LATENT_DIR": join(params, 'LatentParams'),
                  "MODEL_DIR":  join(params, 'Models'),
                  "LOG_DIR":  join(params, 'LogFiles'),
                  "VERSION": default_int(params, "VERSION", 1),
                  "Z_DIM": default_int(params, "Z_DIM", 64),
                  "MIXED_MODEL": default_false(params, "MIXED_MODEL"),
                  "GAN": default_false(params, "GAN"),
                  "IGLS_ITERATIONS": default_int(params, "IGLS_ITERATIONS", 1),
                  "SLOPE": default_false(params, "SLOPE"),
                  "INCLUDE_A01": default_false(params, "INCLUDE_A01"),
                  "SAVE_LATENT": default_false(params, "SAVE_LATENT"),
                  "USE_SAMPLER":  default_false(params, "USE_SAMPLER") or default_false(params, "MIXED_MODEL"),
                  "SAMPLER_PARAMS": [4, 6] if 'SAMPLER_PARAMS' not in params else params['SAMPLER_PARAMS'],
                  "MIN_DATA": default_int(params, "MIN_DATA", 4),
                  "BATCH_SIZE": default_int(params, "BATCH_SIZE", 100),
                  "SHUFFLE_BATCHES": default_false(params, "SHUFFLE_BATCHES"),
                  "H_FLIP": default_float(params, "H_FLIP", 0.),
                  "V_FLIP": default_float(params, "V_FLIP", 0.),
                  "EPOCHS": default_int(params, "EPOCHS", 100),
                  "LR": default_float(params, "LR", 1e-5),
                  "D_LR": default_float(params, "D_LR", 1e-5),
                  "RECON_LOSS": default_true(params, "RECON_LOSS"),
                  "ALIGN_LOSS": default_false(params, "ALIGN_LOSS"),
                  "KL_LOSS": default_false(params, "KL_LOSS"),
                  "D_LOSS": default_false(params, "D_LOSS"),
                  "BETA": default_float(params, "BETA", 1.),
                  "GAMMA": default_float(params, "GAMMA", 1.),
                  "NU": default_float(params, "NU", 1.),
                  "PLOT_EPOCH": default_int(params, "100"),
                  "IMAGE_NO": default_int(params, "IMAGE_NO", 0),
                  "TIMEPOINT": default_int(params, "TIMEPOINT", 0)
                  }
    return param_dict