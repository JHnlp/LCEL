"""
=======================================================================
 Copyright (c) 2019 PolyU CBS LLT Group. All Rights Reserved
@Name       : utils
@Author     : Jinghang GU
@Contect    : gujinghangnlp@gmail.com
@Time       : 2021/8/28 下午1:46

=======================================================================
"""
import os
import random
import torch
import numpy as np


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
