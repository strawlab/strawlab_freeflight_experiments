import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import rospy
import std_msgs.msg

import strawlab_freeflight_experiments.conditions as sfe_conditions
import nodelib.log

import argparse

def get_and_parse_commandline():
    argv = rospy.myargv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-wait', action='store_true', default=False,
                        help="dont't start unless flydra is saving data")
    parser.add_argument('--tmpdir', action='store_true', default=False,
                        help="store logfile in tmpdir")
    parser.add_argument('--continue-existing', type=str, default=None,
                        help="path to a logfile to continue")
    parser.add_argument('--conditions', default=sfe_conditions.get_default_condition_filename(argv),
                        help="path to yaml file experimental conditions")
    parser.add_argument('--start-condition', type=str,
                        help="name of condition to start the experiment with")
    parser.add_argument('--cool-conditions', type=str,
                        help="comma separated list of cool conditions (those for which "\
                             "a video of the trajectory is saved)")
    parser.add_argument('--switch-time', type=int, default=300,
                        help='time in seconds for each condition')
    parser.add_argument('--switch-random', action='store_true', default=False,
                        help='cycle conditions in random order (default is sequential)')

    args = parser.parse_args(argv[1:])

    return parser, args

class Experiment(object):
    def __init__(self, args, state):
        cool_conditions = args.cool_conditions

        self._switch_random = args.switch_random

        self.conditions = sfe_conditions.Conditions(open(args.conditions))
        self.cool_conditions = cool_conditions.split(',') if cool_conditions else set()

        start_condition = args.start_condition if args.start_condition else self.conditions.keys()[0]
        self.condition = self.conditions[start_condition]

        self.log = nodelib.log.CsvLogger(
                          state=state,
                          wait=not args.no_wait, use_tmpdir=args.tmpdir,
                          continue_existing=args.continue_existing)

        self.timer = rospy.Timer(rospy.Duration(args.switch_time),
                                  self._switch_conditions)

        rospy.Subscriber('experiment_uuid',
                         std_msgs.msg.String,
                         self._on_experiment_uuid)

    def _on_experiment_uuid(self, msg):
        self.log.set_experiment_uuid(msg.data)
        #save the condition yaml here

    def run(self):
        raise NotImplementedError

    def switch_conditions(self):
        raise NotImplementedError

    def _switch_conditions(self,event=None):
        if self._switch_random:
            condition = self.conditions.random_condition()
        else:
            condition = self.conditions.next_condition(self.condition)

        rospy.loginfo('condition: %s' % condition.name)

        self.log.condition = condition
        self.condition = condition

        self.switch_conditions()

