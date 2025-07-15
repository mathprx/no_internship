from torch.utils.data import DataLoader
import xarray as xr
from climsim_datasets_neural_operators2 import DatasetNO
from climsim_utils.data_utils_highres import *
from omegaconf import OmegaConf
from tqdm import tqdm

def get_data(cfg: OmegaConf):
    grid_info = xr.open_dataset(cfg.grid_info_path)
    input_mean = xr.open_dataset(cfg.input_mean_path)
    input_max = xr.open_dataset(cfg.input_max_path)
    input_min = xr.open_dataset(cfg.input_min_path)
    output_scale = xr.open_dataset(cfg.output_scale_path)
    qn_lbd = np.loadtxt(cfg.qn_lbd_path, delimiter=',')

    data = data_utils(grid_info = grid_info, 
                    input_mean = input_mean, 
                    input_max = input_max, 
                    input_min = input_min, 
                    output_scale = output_scale)
    
    data.set_to_v2_rh_mc_vars()
    input_scalar_num = data.input_scalar_num
    input_profile_num = data.input_profile_num
    output_scalar_num = data.target_scalar_num
    output_profile_num = data.target_profile_num
    vertical_levels = cfg.learning_vertical_levels
    
    print('Input scalar num:', input_scalar_num)
    print('Input profile num:', input_profile_num)
    print('Output scalar num:', output_scalar_num)
    print('Output profile num:', output_profile_num)
    print('Vertical levels:', vertical_levels)

    input_sub, input_div, out_scale = data.save_norm(write=False)

    train_dataset = DatasetNO(  input_path = cfg.train_input_path,
                                target_path = cfg.train_target_path,
                                input_sub = input_sub,
                                input_div = input_div,
                                out_scale = out_scale,
                                qinput_prune = cfg.qinput_prune,
                                output_prune = cfg.output_prune,
                                strato_lev = cfg.strato_lev,
                                qn_lbd = qn_lbd,
                                decouple_cloud = cfg.decouple_cloud,
                                aggressive_pruning = cfg.aggressive_pruning,
                                strato_lev_qc = cfg.strato_lev_qc,
                                strato_lev_qinput = cfg.strato_lev_qinput,
                                strato_lev_tinput = cfg.strato_lev_tinput,
                                strato_lev_out = cfg.strato_lev_out,
                                input_clip = cfg.input_clip,
                                input_clip_rhonly = cfg.input_clip_rhonly,
                                input_scalar_num = input_scalar_num,
                                input_profile_num = input_profile_num,
                                output_scalar_num = output_scalar_num,
                                output_profile_num = output_profile_num,
                                vertical_levels = vertical_levels,
                                )

    val_dataset = ValidationDatasetNO(val_input_path = cfg.val_input_path,
                                    val_target_path = cfg.val_target_path,
                                    input_sub = input_sub,
                                    input_div = input_div,
                                    out_scale = out_scale,
                                    qinput_prune = cfg.qinput_prune,
                                    output_prune = cfg.output_prune,
                                    strato_lev = cfg.strato_lev,
                                    qn_lbd = qn_lbd,
                                    decouple_cloud = cfg.decouple_cloud,
                                    aggressive_pruning = cfg.aggressive_pruning,
                                    strato_lev_qc = cfg.strato_lev_qc,
                                    strato_lev_qinput = cfg.strato_lev_qinput,
                                    strato_lev_tinput = cfg.strato_lev_tinput,
                                    strato_lev_out = cfg.strato_lev_out,
                                    input_clip = cfg.input_clip,
                                    input_clip_rhonly = cfg.input_clip_rhonly,
                                    input_scalar_num = input_scalar_num,
                                    input_profile_num = input_profile_num,
                                    output_scalar_num = output_scalar_num,
                                    output_profile_num = output_profile_num,
                                    vertical_levels = vertical_levels,
                                    )
    
    print('Number of training samples:', len(train_dataset))
    print('Number of validation samples:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)

    return data, train_loader, val_loader