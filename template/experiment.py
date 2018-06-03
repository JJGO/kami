from datetime import datetime
import json
import os
import pathlib
import signal
import shutil
import subprocess
import sys
import traceback

from api import Data, Model
from kirin.util.cli import color

devnull = open(os.devnull, 'w')

KIRIN_PATH = '/data/ddmg/explog/kirin'


class Experiment():

    def __init__(self, data_params, model_params, train_params, path=None):

        self.args = {'model_params': model_params,
                     'data_params': data_params,
                     'train_params': train_params,
                     }

        self.data, self.data_params = Data(**data_params)

        self.model, self.model_params = Model(input=self.data.shape, **model_params)

        self.train_params = self.model.compile(**train_params)

        self.name = self.semantic_version()
        if path is None:
            path = pathlib.Path.cwd()
        elif not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        self.path = path / 'results' / self.name

        signal.signal(signal.SIGINT, self.SIGINT_handler)
        signal.signal(signal.SIGQUIT, self.SIGQUIT_handler)

    def semantic_version(self):
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        params = {'time': time, **self.model_params, **self.data_params, **self.train_params}
        version = '{time}_{model}_{dataset}'.format(**params)
        return version

    def run(self):
        print(f'\n{color.MAGENTA}Storing results in {self.path}{color.END}\n')
        path = self.path

        path.mkdir(exist_ok=True, parents=True)
        self.path = path

        self.hyperparams = {'model_params': self.model_params,
                            'data_params': self.data_params,
                            'train_params': self.train_params,
                            }

        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
        self.status = {'aborted': False,
                       'failed': False,
                       'commit': commit,
                       'pid': os.getpid()
                       }
        print(self.hyperparams)
        self.save_json('args', self.args)
        self.save_json('hyperparams', self.hyperparams)
        self.save_json('status', self.status)

        # Save Source code
        subprocess.call(['zip', '-r', (path / 'source.zip').as_posix(), '.', '-i', '*.py'], stdout=devnull, stderr=devnull)
        subprocess.call(['zip', '-r', (path / 'kirin.zip').as_posix(), KIRIN_PATH, '-i', '*.py'], stdout=devnull, stderr=devnull)

        self.train_params = {'path': path, **self.train_params}
        try:
            self.model.fit(self.data, path)
        except Exception as e:
            self.status['failed'] = True
            self.save_json('status', self.status)
            traceback.print_exc()

    def save_json(self, name, json_dict):
        with open(self.path / f'{name}.json', 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=True)

    def SIGINT_handler(self, signal, frame):
        self.status['aborted'] = True
        self.save_json('status', self.status)
        sys.exit(1)

    def SIGQUIT_handler(self, signal, frame):
        shutil.rmtree(self.path, ignore_errors=True)
        sys.exit(1)
