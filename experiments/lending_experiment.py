# coding=utf-8
# Copyright 2020 The ML Fairness Gym Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main file to run lending experiments for demonstration purposes."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
from absl import app
from absl import flags
from agents import threshold_policies
from experiments import lending
from experiments import lending_plots
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json

flags.DEFINE_string('plots_directory', None, 'Directory to write out plots.')
flags.DEFINE_bool('equalize_opportunity', False,
                  'If true, apply equality of opportunity constraints.')
flags.DEFINE_integer('num_steps', 10000,
                     'Number of steps to run the simulation.')

FLAGS = flags.FLAGS

# Control float precision in json encoding.
json.encoder.FLOAT_REPR = lambda o: repr(round(o, 3))

MAXIMIZE_REWARD = threshold_policies.ThresholdPolicy.MAXIMIZE_REWARD
EQUALIZE_OPPORTUNITY = threshold_policies.ThresholdPolicy.EQUALIZE_OPPORTUNITY


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    np.random.seed(100)
    group_0_prob = 0.5
    maximize_reward_result = lending.Experiment(
        group_0_prob=group_0_prob,
        interest_rate=1.0,
        bank_starting_cash=10000,
        seed=200,
        num_steps=FLAGS.num_steps,
        burnin=200,
        cluster_shift_increment=0.01,
        include_cumulative_loans=True,
        return_json=False,
        threshold_policy=MAXIMIZE_REWARD).run()
    equality_of_opportunity_result = lending.Experiment(
        group_0_prob=group_0_prob,
        interest_rate=1.0,
        bank_starting_cash=10000,
        seed=200,
        num_steps=FLAGS.num_steps,
        burnin=200,
        cluster_shift_increment=0.01,
        include_cumulative_loans=True,
        return_json=False,
        threshold_policy=EQUALIZE_OPPORTUNITY).run()

    lending_plots.do_plotting(maximize_reward_result,
                  equality_of_opportunity_result,
                  equality_of_opportunity_result,
                  # static_equality_of_opportunity_result,
                  FLAGS.plots_directory,
                  options=None)


if __name__ == '__main__':
  app.run(main)
