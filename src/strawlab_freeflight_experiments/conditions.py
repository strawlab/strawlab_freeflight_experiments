import warnings
import os

if int(os.environ.get('STRAWLAB_THROW_DEPRECATION', '0')):
    raise RuntimeError('you are importing a deprecated module')

warnings.warn("You are importing module "
              "'strawlab_freeflight_experiments.conditions'. Please update this "
              "to module 'freeflight_analysis.conditions'")

from freeflight_analysis.conditions import *
