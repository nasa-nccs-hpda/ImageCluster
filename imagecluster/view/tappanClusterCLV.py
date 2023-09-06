import argparse
import omegaconf
import pathlib
import sys

from imagecluster.model.TappanClusterer import TappanClusterer


def parseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c',
                        '--config',
                        dest='config',
                        type=str,
                        required=True,
                        help='Path to YAML configuration file')

    args = parser.parse_args()

    return args


def main() -> int:

    args = parseArgs()

    config_file = pathlib.Path(args.config)

    if not config_file.exists():

        error_msg = 'Need a config file that exists. ' + \
            f'{config_file} does not exist.'

        raise FileNotFoundError(error_msg)

    conf = omegaconf.OmegaConf.load(config_file)

    tappanCluster = TappanClusterer(conf)

    tappanCluster.run()


if __name__ == '__main__':

    sys.exit(main())
