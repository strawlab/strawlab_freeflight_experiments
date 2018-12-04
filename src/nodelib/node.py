import os.path
import argparse
import yaml

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
roslib.load_manifest('flycave')

import rospy
import std_msgs.msg
import flycave.msg

import freeflight_analysis.conditions as sfe_conditions
import nodelib.log

def get_and_parse_commandline():
    argv = rospy.myargv()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('--max-number-cool-conditions', type=int, default=0,
                        help='collect a maximum number of cool conditions (0 disables)')
    parser.add_argument('--switch-time', type=int, default=300,
                        help='time in seconds for each condition')
    parser.add_argument('--switch-order', nargs='?', choices=['seq', 'randstart', 'fullrand'], default='seq',
                        help='Controls the order of condition switch during the experiment. Options:\n'
                             ' - seq: keep keep the order in the yaml file\n'
                             ' - randstart: use always the same, randomized order\n'
                             ' - rand: randomize the order continuously\n')
    parser.add_argument('--switch-seed', type=int, default=42,
                        help='The random seed used to control condition order randomization.'
                             'If negative, then randomization is based on the system clock.')

    args = parser.parse_args(argv[1:])

    return parser, args


class Experiment(object):
    def __init__(self, args, state):
        cool_conditions = args.cool_conditions

        # N.B. randomisation info can be queried at a later time in self.conditions
        self.conditions = sfe_conditions.Conditions(open(args.conditions),
                                                    rng_seed=args.switch_seed,
                                                    switch_order=args.switch_order)
        
        self._n_cool = 0
        self._max_cool = args.max_number_cool_conditions
        self.cool_conditions = cool_conditions.split(',') if cool_conditions else set()

        for c in self.cool_conditions:
            if c not in self.conditions:
                rospy.logwarn("cool condition %s does not exist" % c)

        start_condition = args.start_condition if args.start_condition else self.conditions.keys()[0]
        self.condition = self.conditions[start_condition]

        self.log = nodelib.log.CsvLogger(
                          state=state,
                          wait=not args.no_wait, use_tmpdir=args.tmpdir,
                          continue_existing=args.continue_existing)
        self.log.condition = self.condition

        self.timer = rospy.Timer(rospy.Duration(args.switch_time),
                                  self._switch_conditions)

        rospy.Subscriber('experiment',
                         flycave.msg.Experiment,
                         self._on_experiment)

        self.pub_pushover = rospy.Publisher('note', std_msgs.msg.String)
        self.pub_save = rospy.Publisher('save_object', std_msgs.msg.UInt32)

    def _on_experiment(self, msg):
        self.log.set_experiment_uuid(msg.uuid)
        path,fname = os.path.split(self.log._fname)
        date_fname = fname.split('.')[0]

        #save the condition yaml
        with open(os.path.join(path,date_fname)+'.condition.yaml','w') as f:
            f.write(self.conditions.to_yaml())
            f.write("uuid: %s\n" % msg.uuid)

        exp = {}
        for i in msg.__slots__:
            v = getattr(msg,i)
            if not isinstance(v,(int,str,bool,float,unicode)):
                v = str(v)
            exp[i] = v
        with open(os.path.join(path,date_fname)+'.experiment.yaml','w') as f:
            yaml.safe_dump(exp, f, default_flow_style=False)

    def run(self):
        raise NotImplementedError

    def switch_conditions(self):
        raise NotImplementedError

    def save_cool_condition(self, obj_id, note=''):
        if self.condition.name in self.cool_conditions:
            if self._n_cool < self._max_cool:
                if not note:
                    note = "Subject %d met cool condition"

                note += "(condition: '%s')" % self.condition.name

                self.pub_pushover.publish(note)
                self.pub_save.publish(obj_id)
                self._n_cool += 1

                rospy.loginfo('cool: %s' % note)

    def _switch_conditions(self,event=None):        

        condition = self.conditions.next_condition(self.condition)

        rospy.logdebug('condition: %s' % condition.name)

        self.log.condition = condition
        self.condition = condition

        self.switch_conditions()
