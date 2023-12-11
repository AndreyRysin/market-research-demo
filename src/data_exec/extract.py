from argparse import ArgumentParser

import _init

from configs.data_configs import init_data_config
from data_modules.extraction_wrapper import ExtractionWrapper


if __name__ == '__main__':
    # init args
    argparser = ArgumentParser()
    argparser.add_argument('--mode', dest='mode', help='run for features or targets', required=True)
    argparser.add_argument('--data_cfg', dest='data_cfg', help='data config', required=True)
    args = argparser.parse_args()
    if not (args.mode in ['features', 'targets']):
        raise ValueError(f'option `mode` must be "features" or "targets" but "{args.mode}" is given')
    # extract
    ew = ExtractionWrapper(args.mode)
    ew.set_config(init_data_config(args.data_cfg))
    ew.load_metadata()
    ew.open_storage()
    ew.extract()
    ew.close_storage()
