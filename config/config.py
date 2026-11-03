import configparser
import os


# Maps the friendly --data name to its diffusion config file (relative to project root).
DATA_CONFIGS = {
    'COMA': 'config/config_diffusion.cfg',
    'DFAUST': 'config/config_diffusion_dfaust.cfg',
}


def resolve_config_file(data=None, config_file=None):
    """Resolve which config file to use.

    An explicit --config_file always wins. Otherwise --data (COMA/DFAUST) selects
    the matching diffusion config. Returns an absolute path.
    """
    if config_file:
        return config_file
    if data is None:
        raise ValueError("Provide either --config_file or --data (COMA|DFAUST).")
    key = data.upper()
    if key not in DATA_CONFIGS:
        raise ValueError(f"Unknown --data '{data}'. Choose one of {list(DATA_CONFIGS)}.")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(project_root, DATA_CONFIGS[key].replace('/', os.sep))


def read_config(file_name):
    config = configparser.RawConfigParser()
    config.read(file_name)

    config_parms = {}

    dataset_dir = config.get('I/O parameters', 'dataset_dir')
    template_file = config.get('I/O parameters', 'template_file')
    checkpoint_dir = config.get('I/O parameters', 'checkpoint_dir')

    # Resolve PATH_TO_PROJECT placeholder: use directory containing the config folder as project root
    if 'PATH_TO_PROJECT' in dataset_dir or 'PATH_TO_PROJECT' in template_file or 'PATH_TO_PROJECT' in checkpoint_dir:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dataset_dir = dataset_dir.replace('PATH_TO_PROJECT', project_root).replace('/', os.sep)
        template_file = template_file.replace('PATH_TO_PROJECT', project_root).replace('/', os.sep)
        checkpoint_dir = checkpoint_dir.replace('PATH_TO_PROJECT', project_root).replace('/', os.sep)

    config_parms['dataset_dir'] = dataset_dir
    config_parms['template_file'] = template_file
    config_parms['checkpoint_dir'] = checkpoint_dir

    # Optional: read raw training meshes from somewhere other than
    # <dataset_dir>/train, so a run can keep its processed cache, template and
    # checkpoints in a separate root without copying the meshes.
    train_dir = config.get('I/O parameters', 'train_dir', fallback='')
    if train_dir:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        train_dir = train_dir.replace('PATH_TO_PROJECT', project_root).replace('/', os.sep)
        config_parms['train_dir'] = train_dir

    config_parms['n_layers'] = config.getint('model parameters', 'n_layers')
    config_parms['z_length'] = config.getint('model parameters', 'z_length')
    config_parms['down_sampling_factors'] = [int(x) for x in
                                             config.get('model parameters', 'down_sampling_factors').split(',')]
    config_parms['num_features_global'] = [int(x) for x in
                                           config.get('model parameters', 'num_features_global').split(',')]
    config_parms['num_features_local'] = [int(x) for x in
                                          config.get('model parameters', 'num_features_local').split(',')]
    config_parms['batch_norm'] = True if config.getint('model parameters', 'batch_norm') == 1 else False
    config_parms['model_type'] = config.get('model parameters', 'model_type', fallback='default')
    # Optional: input channels for completion (default = num_features_global[0], i.e. 3)
    config_parms['in_channels'] = config.getint('model parameters', 'in_channels',
                                                 fallback=config_parms['num_features_global'][0])
    # Input block type: 'linear' (V1 simple projection) or 'graph_aware' (V3 with FeaStConv)
    config_parms['input_block_type'] = config.get('model parameters', 'input_block_type', fallback='linear')
    config_parms['input_block_hidden'] = config.getint('model parameters', 'input_block_hidden', fallback=16)

    config_parms['num_workers'] = config.getint('training parameters', 'num_workers')
    config_parms['lr'] = config.getfloat('training parameters', 'lr')
    config_parms['batch_size'] = config.getint('training parameters', 'batch_size')
    config_parms['weight_decay'] = config.getfloat('training parameters', 'weight_decay')
    config_parms['epoch'] = config.getint('training parameters', 'epoch')
    config_parms["lambda_reg"] = config.getfloat("training parameters", "lambda_reg")

    # New parameter for experiment name
    config_parms['experiment_name'] = config.get('training parameters', 'experiment_name', fallback='default_experiment')
    
    #########  princlple
    # config_parms['include_curvature'] = config.getint('model parameters', 'include_curvature') == 1

    # ---- Completion-specific parameters (optional section) ----
    if config.has_section('completion parameters'):
        config_parms['mask_ratio_min'] = config.getfloat('completion parameters', 'mask_ratio_min', fallback=0.10)
        config_parms['mask_ratio_max'] = config.getfloat('completion parameters', 'mask_ratio_max', fallback=0.40)
        config_parms['lambda_missing'] = config.getfloat('completion parameters', 'lambda_missing', fallback=3.0)
        config_parms['lambda_known'] = config.getfloat('completion parameters', 'lambda_known', fallback=1.0)

        # Geodesic depth-ramped loss weighting (Olivier, 07-25 thread): replace
        # the flat lambda_missing over the hole with
        #   w(v) = lambda_known + (w_max - lambda_known) * d(v)/d_max
        # where d(v) is the geodesic distance from missing vertex v to the
        # known boundary. Deep = hard = upweighted. Default OFF, so every
        # existing run reproduces byte-for-byte.
        config_parms['depth_weight'] = config.getint(
            'completion parameters', 'depth_weight', fallback=0) == 1
        config_parms['depth_weight_max'] = config.getfloat(
            'completion parameters', 'depth_weight_max',
            fallback=config_parms['lambda_missing'])
        config_parms['depth_weight_backend'] = config.get(
            'completion parameters', 'depth_weight_backend', fallback='dijkstra')
        # Rescale the ramp so its MEAN over the hole equals depth_weight_max.
        # The raw ramp averages ~w_max/2, so without this the arm changes both
        # the shape of the weighting AND the overall missing-vs-known emphasis.
        config_parms['depth_weight_normalize'] = config.getint(
            'completion parameters', 'depth_weight_normalize', fallback=0) == 1
        config_parms['pretrained_encoder'] = config.get('completion parameters', 'pretrained_encoder', fallback='')
        config_parms['pretrained_decoder'] = config.get('completion parameters', 'pretrained_decoder', fallback='')

        # Which masking strategies to use and their sampling weights. For bodies we
        # want CONTIGUOUS holes (a missing limb/region), so DFAUST drops the
        # scatter strategy. Defaults preserve the original COMA behaviour.
        strat = config.get('completion parameters', 'mask_strategies',
                           fallback='bfs_patch,plane_cut,random_scatter')
        config_parms['mask_strategies'] = tuple(s.strip() for s in strat.split(',') if s.strip())
        wts = config.get('completion parameters', 'mask_strategy_weights',
                         fallback='0.5,0.3,0.2')
        config_parms['mask_strategy_weights'] = tuple(float(x) for x in wts.split(','))

        # V2: Chamfer distance loss
        config_parms['use_chamfer'] = config.getint('completion parameters', 'use_chamfer', fallback=0) == 1
        config_parms['lambda_chamfer'] = config.getfloat('completion parameters', 'lambda_chamfer', fallback=1.0)

        # V2: Curriculum learning (gradually increasing mask ratio)
        config_parms['curriculum'] = config.getint('completion parameters', 'curriculum', fallback=0) == 1
        config_parms['curriculum_start_min'] = config.getfloat('completion parameters', 'curriculum_start_min', fallback=0.02)
        config_parms['curriculum_start_max'] = config.getfloat('completion parameters', 'curriculum_start_max', fallback=0.05)
        config_parms['curriculum_end_min'] = config.getfloat('completion parameters', 'curriculum_end_min', fallback=0.10)
        config_parms['curriculum_end_max'] = config.getfloat('completion parameters', 'curriculum_end_max', fallback=0.15)

    assert config_parms['n_layers'] == len(
        config_parms['down_sampling_factors']), 'length of down_sampling_factors must equal to n_layers'
    assert config_parms['n_layers'] + 1 == len(
        config_parms['num_features_global']), 'length of num_features must equal to n_layers + 1'

    return config_parms
