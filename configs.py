import numpy as np
import torch
import torch.nn as nn
import math

GRAM64_FFHQ = {
    'global': {
        'img_size': 64,
        'batch_size': 8,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 1,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 1.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 64,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*0.5,
            'sample_dist': 'gaussian',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderDiscriminator',
        'kwargs': {
            'img_size': 64,
        }
    },
    'dataset': {
        'class': 'FFHQ',
        'kwargs': {
            'img_size': 64,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}

GRAM64_AFHQ = {
    'global': {
        'img_size': 64,
        'batch_size': 8,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 1,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 1.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 64,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*0.5,
            'sample_dist': 'gaussian',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderDiscriminator',
        'kwargs': {
            'img_size': 64,
        }
    },
    'dataset': {
        'class': 'AFHQCats',
        'kwargs': {
            'img_size': 64,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}


GRAM64_CARLA = {
    'global': {
        'img_size': 64,
        'batch_size': 4,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        # 'sampling_network_lr': 2e-6,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 4,
            'real_pos_lambda': 15.,
            'r1_lambda': 1.,
            'pos_lambda': 15.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 64,
            'h_stddev': math.pi,
            'v_stddev': math.pi*(42.5/180),
            'h_mean': math.pi*0.5,
            'v_mean': math.pi*(42.5/180),
            'sample_dist': 'spherical_uniform',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                # 'normalize': 2,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 256,
                'layer_num_sample': 3,
                'center': (0, 0, 0),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 48,
                'levels_start': 35,
                'levels_end': 5,
                'delta_alpha': 0.02,
                'last_back': False,
                'white_back': True,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderDiscriminator',
        'kwargs': {
            'img_size': 64,
        }
    },
    'dataset': {
        'class': 'CARLA',
        'kwargs': {
            'img_size': 64,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 30,
        'ray_start': 0.7,
        'ray_end': 1.3,
    }
}

GRAM64_CATS = {
    'global': {
        'img_size': 64,
        'batch_size': 8,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 1,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 1.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 64,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*0.5,
            'sample_dist': 'gaussian',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderDiscriminator',
        'kwargs': {
            'img_size': 64,
        }
    },
    'dataset': {
        'class': 'Cats',
        'kwargs': {
            'img_size': 64,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}


GRAMHD256_FFHQ = {
    'global': {
        'img_size': 256,
        'batch_size': 1,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-4,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'SRGan3DProcess',
        'kwargs': {
            'batch_split': 1,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 3.,
            'cons_lambda': 1,
            'use_patch_d': True,
            'patch_lambda': 0.1,
            'r1_patch': True,
        }
    },
    'generator': {
        'class': 'GramHDGenerator',
        'kwargs': {
            'z_dim': 256,
            'feature_dim': 32, 
            'img_size': 256,
            'lr_img_size': 64,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*0.5,
            'sample_dist': 'gaussian',
            'gram_model_file': './training_runs/ffhq64/generator.pth',    # If you want to train your own model, set this to the stage1 GRAM model file
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'super_resolution': {
            'class': 'styleesrgan',
            'kwargs': {
                'fg': {
                    'w_dim': 256,
                    'nf': 64, 
                    'nb': 8,
                    'gc': 32,
                    'up_channels': [64, 64],
                    'to_rgb_ks': 1,
                },
                'bg': {
                    'nf': 64, 
                    'nb': 4,
                    'gc': 32,
                    'up_channels': [64, 32],
                    'use_pixel_shuffle': False,
                    'global_residual': True
                },
            }
        },
        'renderer': {
            'class': 'manifold_sr_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderPatchDiscriminator',
        'kwargs': {
            'img_size': 256,
            'norm_layer': nn.Identity,
        }
    },
    'dataset': {
        'class': 'FFHQ',
        'kwargs': {
            'img_size': 256,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}

GRAMHD256_AFHQ = {
    'global': {
        'img_size': 256,
        'batch_size': 4,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-4,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'SRGan3DProcess',
        'kwargs': {
            'batch_split': 4,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 3.,
            'cons_lambda': 1,
            'use_patch_d': True,
            'patch_lambda': 0.1,
            'r1_patch': True,
        }
    },
    'generator': {
        'class': 'GramHDGenerator',
        'kwargs': {
            'z_dim': 256,
            'feature_dim': 32, 
            'img_size': 256,
            'lr_img_size': 64,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*0.5,
            'sample_dist': 'gaussian',
            'gram_model_file': None,    # If you want to train your own model, set this to the stage1 GRAM model file
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'super_resolution': {
            'class': 'styleesrgan',
            'kwargs': {
                'fg': {
                    'w_dim': 256,
                    'nf': 64, 
                    'nb': 8,
                    'gc': 32,
                    'up_channels': [64, 64],
                    'to_rgb_ks': 1,
                },
                'bg': {
                    'nf': 64, 
                    'nb': 4,
                    'gc': 32,
                    'up_channels': [64, 32],
                    'use_pixel_shuffle': False,
                    'global_residual': True
                },
            }
        },
        'renderer': {
            'class': 'manifold_sr_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderPatchDiscriminator',
        'kwargs': {
            'img_size': 256,
            'norm_layer': nn.Identity,
        }
    },
    'dataset': {
        'class': 'AFHQCats',
        'kwargs': {
            'img_size': 256,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}

GRAMHD256_CARLA = {
    'global': {
        'img_size': 256,
        'batch_size': 1,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-4,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'SRGan3DProcess',
        'kwargs': {
            'batch_split': 1,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 3.,
            'cons_lambda': 1,
            'use_patch_d': True,
            'patch_lambda': 0.1,
            'r1_patch': True,
        }
    },
    'generator': {
        'class': 'GramHDGenerator',
        'kwargs': {
            'z_dim': 256,
            'feature_dim': 32, 
            'img_size': 256,
            'lr_img_size': 64,
            'h_stddev': np.pi,
            'v_stddev': np.pi*(42.5/180),
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*(42.5/180),
            'sample_dist': 'spherical_uniform',
            'gram_model_file': None,    # If you want to train your own model, set this to the stage1 GRAM model file
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'super_resolution': {
            'class': 'styleesrgan',
            'kwargs': {
                'fg': {
                    'w_dim': 256,
                    'nf': 64, 
                    'nb': 8,
                    'gc': 32,
                    'up_channels': [64, 64],
                    'to_rgb_ks': 1,
                },
                'bg': {
                    'nf': 64, 
                    'nb': 4,
                    'gc': 32,
                    'up_channels': [64, 32],
                    'use_pixel_shuffle': False,
                    'global_residual': True
                },
            }
        },
        'renderer': {
            'class': 'manifold_sr_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderPatchDiscriminator',
        'kwargs': {
            'img_size': 256,
            'norm_layer': nn.Identity,
        }
    },
    'dataset': {
        'class': 'CARLA',
        'kwargs': {
            'img_size': 256,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 30,
        'ray_start': 0.7,
        'ray_end': 1.3,
    }
}


GRAMHD256_CATS = {
    'global': {
        'img_size': 256,
        'batch_size': 4,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-4,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'SRGan3DProcess',
        'kwargs': {
            'batch_split': 4,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 3.,
            'cons_lambda': 1,
            'use_patch_d': True,
            'patch_lambda': 0.1,
            'r1_patch': True,
        }
    },
    'generator': {
        'class': 'GramHDGenerator',
        'kwargs': {
            'z_dim': 256,
            'feature_dim': 32, 
            'img_size': 256,
            'lr_img_size': 64,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*0.5,
            'sample_dist': 'gaussian',
            'gram_model_file': './training_runs/cats64/step100000_generator.pth',    # If you want to train your own model, set this to the stage1 GRAM model file
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'super_resolution': {
            'class': 'styleesrgan',
            'kwargs': {
                'fg': {
                    'w_dim': 256,
                    'nf': 64, 
                    'nb': 8,
                    'gc': 32,
                    'up_channels': [64, 64],
                    'to_rgb_ks': 1,
                },
                'bg': {
                    'nf': 64, 
                    'nb': 4,
                    'gc': 32,
                    'up_channels': [64, 32],
                    'use_pixel_shuffle': False,
                    'global_residual': True
                },
            }
        },
        'renderer': {
            'class': 'manifold_sr_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderPatchDiscriminator',
        'kwargs': {
            'img_size': 256,
            'norm_layer': nn.Identity,
        }
    },
    'dataset': {
        'class': 'Cats',
        'kwargs': {
            'img_size': 256,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}


GRAMHD512_FFHQ = {
    'global': {
        'img_size': 512,
        'batch_size': 1,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-4,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'SRGan3DProcess',
        'kwargs': {
            'batch_split': 1,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 5.,
            'cons_lambda': 3.,
            'use_patch_d': True,
            'patch_lambda': 0.1,
            'r1_patch': True,
        }
    },
    'generator': {
        'class': 'GramHDGenerator',
        'kwargs': {
            'z_dim': 256,
            'feature_dim': 32, 
            'img_size': 512,
            'lr_img_size': 64,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*0.5,
            'sample_dist': 'gaussian',
            'gram_model_file': './training_runs/ffhq64/generator.pth',    # If you want to train your own model, set this to the stage1 GRAM model file
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'super_resolution': {
            'class': 'styleesrgan',
            'kwargs': {
                'fg': {
                    'w_dim': 256,
                    'nf': 64, 
                    'nb': 8,
                    'gc': 32,
                    'up_channels': [64, 64, 32],
                    'to_rgb_ks': 1,
                },
                'bg': {
                    'nf': 64,
                    'nb': 4,
                    'gc': 32,
                    'up_channels': [64, 32, 16],
                    'use_pixel_shuffle': False,
                    'global_residual': True
                },
            }
        },
        'renderer': {
            'class': 'manifold_sr_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderPatchDiscriminator',
        'kwargs': {
            'img_size': 512,
            'norm_layer': nn.Identity,
        }
    },
    'dataset': {
        'class': 'FFHQ',
        'kwargs': {
            'img_size': 512,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}

GRAMHD512_AFHQ = {
    'global': {
        'img_size': 512,
        'batch_size': 2,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-4,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'SRGan3DProcess',
        'kwargs': {
            'batch_split': 2,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 5.,
            'cons_lambda': 3.,
            'use_patch_d': True,
            'patch_lambda': 0.1,
            'r1_patch': True,
        }
    },
    'generator': {
        'class': 'GramHDGenerator',
        'kwargs': {
            'z_dim': 256,
            'feature_dim': 32, 
            'img_size': 512,
            'lr_img_size': 64,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*0.5,
            'sample_dist': 'gaussian',
            'gram_model_file': None,    # If you want to train your own model, set this to the stage1 GRAM model file
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'super_resolution': {
            'class': 'styleesrgan',
            'kwargs': {
                'fg': {
                    'w_dim': 256,
                    'nf': 64, 
                    'nb': 8,
                    'gc': 32,
                    'up_channels': [64, 64, 32],
                    'to_rgb_ks': 1,
                },
                'bg': {
                    'nf': 64,
                    'nb': 4,
                    'gc': 32,
                    'up_channels': [64, 32, 16],
                    'use_pixel_shuffle': False,
                    'global_residual': True
                },
            }
        },
        'renderer': {
            'class': 'manifold_sr_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderPatchDiscriminator',
        'kwargs': {
            'img_size': 512,
            'norm_layer': nn.Identity,
        }
    },
    'dataset': {
        'class': 'AFHQCats',
        'kwargs': {
            'img_size': 512,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}

GRAMHD512_CARLA = {
    'global': {
        'img_size': 512,
        'batch_size': 2,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-4,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'SRGan3DProcess',
        'kwargs': {
            'batch_split': 2,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 5.,
            'cons_lambda': 3.,
            'use_patch_d': True,
            'patch_lambda': 0.1,
            'r1_patch': True,
        }
    },
    'generator': {
        'class': 'GramHDGenerator',
        'kwargs': {
            'z_dim': 256,
            'feature_dim': 32, 
            'img_size': 512,
            'lr_img_size': 64,
            'h_stddev': np.pi,
            'v_stddev': np.pi*(42.5/180),
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*(42.5/180),
            'sample_dist': 'spherical_uniform',
            'gram_model_file': None,    # If you want to train your own model, set this to the stage1 GRAM model file
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'super_resolution': {
            'class': 'styleesrgan',
            'kwargs': {
                'fg': {
                    'w_dim': 256,
                    'nf': 64, 
                    'nb': 8,
                    'gc': 32,
                    'up_channels': [64, 64, 32],
                    'to_rgb_ks': 1,
                },
                'bg': {
                    'nf': 64,
                    'nb': 4,
                    'gc': 32,
                    'up_channels': [64, 32, 16],
                    'use_pixel_shuffle': False,
                    'global_residual': True
                },
            }
        },
        'renderer': {
            'class': 'manifold_sr_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderPatchDiscriminator',
        'kwargs': {
            'img_size': 512,
            'norm_layer': nn.Identity,
        }
    },
    'dataset': {
        'class': 'CARLA',
        'kwargs': {
            'img_size': 512,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 30,
        'ray_start': 0.7,
        'ray_end': 1.3,
    }
}

GRAMHD512_Cats = {
    'global': {
        'img_size': 512,
        'batch_size': 2,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-4,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'SRGan3DProcess',
        'kwargs': {
            'batch_split': 2,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 5.,
            'cons_lambda': 3.,
            'use_patch_d': True,
            'patch_lambda': 0.1,
            'r1_patch': True,
        }
    },
    'generator': {
        'class': 'GramHDGenerator',
        'kwargs': {
            'z_dim': 256,
            'feature_dim': 32, 
            'img_size': 512,
            'lr_img_size': 64,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*0.5,
            'sample_dist': 'gaussian',
            'gram_model_file': None,    # If you want to train your own model, set this to the stage1 GRAM model file
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'super_resolution': {
            'class': 'styleesrgan',
            'kwargs': {
                'fg': {
                    'w_dim': 256,
                    'nf': 64, 
                    'nb': 8,
                    'gc': 32,
                    'up_channels': [64, 64, 32],
                    'to_rgb_ks': 1,
                },
                'bg': {
                    'nf': 64,
                    'nb': 4,
                    'gc': 32,
                    'up_channels': [64, 32, 16],
                    'use_pixel_shuffle': False,
                    'global_residual': True
                },
            }
        },
        'renderer': {
            'class': 'manifold_sr_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderPatchDiscriminator',
        'kwargs': {
            'img_size': 512,
            'norm_layer': nn.Identity,
        }
    },
    'dataset': {
        'class': 'Cats',
        'kwargs': {
            'img_size': 512,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}

GRAMHD1024_FFHQ = {
    'global': {
        'img_size': 1024,
        'batch_size': 1,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-4,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'SRGan3DProcess',
        'kwargs': {
            'batch_split': 1,
            'pos_lambda': 15.,
            'real_pos_lambda': 15.,
            'r1_lambda': 10.,
            'cons_lambda': 10,
            'use_patch_d': True,
            'patch_lambda': 0.1,
            'r1_patch': True,
        }
    },
    'generator': {
        'class': 'GramHDGenerator',
        'kwargs': {
            'z_dim': 256,
            'feature_dim': 32, 
            'img_size': 1024,
            'lr_img_size': 64,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': np.pi*0.5,
            'v_mean': np.pi*0.5,
            'sample_dist': 'gaussian',
            'gram_model_file': './training_runs/ffhq64/generator.pth',    # If you want to train your own model, set this to the stage1 GRAM model file
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'super_resolution': {
            'class': 'styleesrgan',
            'kwargs': {
                'fg': {
                    'w_dim': 256,
                    'nf': 64, 
                    'nb': 8,
                    'gc': 32,
                    'up_channels': [64, 64, 32, 16],
                    'to_rgb_ks': 1,
                },
                'bg': {
                    'nf': 64,
                    'nb': 4,
                    'gc': 32,
                    'up_channels': [64, 32, 16, 8],
                    'use_pixel_shuffle': False,
                    'global_residual': True
                },
            }
        },
        'renderer': {
            'class': 'manifold_sr_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
            }
        }
    },
    'discriminator': {
        'class': 'GramEncoderPatchDiscriminator',
        'kwargs': {
            'img_size': 1024,
            'norm_layer': nn.Identity,
        }
    },
    'dataset': {
        'class': 'FFHQ',
        'kwargs': {
            'img_size': 1024,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}
