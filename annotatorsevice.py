#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xLiMe Annotator
Luis Rei <luis.rei@ijs.si> @lmrei http://luisrei.com
Feb 2016

To terminate press CONTROL + C or:

    kill -s INT <pid>

"""

import os
import ConfigParser
import argparse
import multiprocessing
import logging
from functools import partial
from zmqservice import serve, worker_task_builder
from annotator import process_message, create_router


DEFAULT_CONFIG = 'annotator.cfg'
ENV_PREFIX = 'ANNOTATOR_'
DEFAULT_DIR = '/etc/annotator/'
DEFAULT_PORT = 1984
DEFAULT_WORKERS = multiprocessing.cpu_count()
DEFAULT_BACKEND = 'ipc://annotbackend.ipc'
DEFAULT_LOG = '/tmp/annotator.log'
DEFAULT_LOGLEVEL = logging.DEBUG


def init_config():
    """Setup configuration defaults
    """
    defaults = {'backend': DEFAULT_BACKEND,
                'workers': DEFAULT_WORKERS,
                'port': DEFAULT_PORT,
                'log': DEFAULT_LOG,
                'loglevel': DEFAULT_LOGLEVEL,
                'ngrams': 4,
                'ngrams_out': False,
                'preprocessor': 'twokenizer',
                'normalizer_type': 'basic',
                'sentiment_out': False,
                'ner_type': 'stanford',
                'ner_out': False,
                'pos_type': 'stanford',
                'pos_out': False
                }

    config = ConfigParser.RawConfigParser(defaults=defaults)
    config.add_section('service')

    # Service PORT
    config.set('service', 'port', DEFAULT_PORT)

    # Backend address
    config.set('service', 'backend', DEFAULT_BACKEND)

    # Number of workers
    config.set('service', 'workers', DEFAULT_WORKERS)

    # logging
    config.set('service', 'log', DEFAULT_LOG)
    config.set('service', 'loglevel', DEFAULT_LOGLEVEL)

    return config


def read_config_file(config, filepath=None):
    """Reads configuration from a file. Searches paths for file
    """
    # List of possible configuration paths
    confpaths = []

    # Add argument filepath
    if filepath is not None:
        confpaths.append(filepath)

    # Add environment filepath
    envpath = os.getenv(ENV_PREFIX+'CONFIG', None)
    if envpath is not None:
        confpaths.append(fenvpath)

    # Add default dir
    confpaths.append(os.path.join(DEFAULT_DIR, DEFAULT_CONFIG))

    # Home
    confpaths.append(os.path.join(os.path.expanduser('~/'), 
                                  '.' + DEFAULT_CONFIG))

    # current dir
    confpaths.append(os.path.join('./', DEFAULT_CONFIG))


    # Try each path until one returns
    for fpath in confpaths:
        if fpath is not None and os.path.isfile(fpath):
            try:
                config.read(fpath)
                m = 'Reading config from {}'.format(fpath)
                print(m)
                return config
            except:
                pass

    # If none return, just spit the default back out
    return config


def setup_logging(config):
    """Sets up logging
    """
    format_str = "%(asctime)-15s %(process)d: %(message)s"
    logpath = config.get('service', 'log')
    loglevel = config.getint('service', 'loglevel')

    # General logging
    logging.basicConfig(filename=logpath, format=format_str, level=loglevel)


def save_config(config, filepath):
    """Saves the current configuration to a file
    """
    with open(filepath, 'wb') as fout:
        config.write(fout)


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Run Canonical URL Service.')

    parser.add_argument('--port', type=int, default=0,
                        help='read/write to zmq socket at specified port')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of concurrent workers')
    parser.add_argument('--config', type=str, default=None,
                        help='configuration file')
    parser.add_argument('--save-config', type=str, default=None,
                        help='Export configuration to this file')

    # Parse
    args = parser.parse_args()

    # Defaul Config
    config = init_config()

    # Load File
    config = read_config_file(config, filepath=args.config)

    if args.port > 0:
        config.set('service', 'port', args.port)
    if args.workers > 0:
        config.set('service', 'workers', args.workers)

    # get final options
    port = config.get('service', 'port')
    n_workers = config.getint('service', 'workers')
    backend = config.get('service', 'backend')

    # Save config
    if args.save_config is not None:
        save_config(config, args.save_config)
    
    # Setup logging
    setup_logging(config)

    # create router
    router, outputs = create_router(config)

    # Setup worker function
    f = partial(process_message, router=router, outputs=outputs)
    worker_task = worker_task_builder(f, backend)
    
    # Print PID
    m = 'Starting Annotator Service with PID: {}'.format(os.getpid())
    logging.info(m)
    print(m)

    # Run forever (or until kill -INT)
    serve(port, worker_task, n_workers, backend)


if __name__ == '__main__':
    main()
