import os.path
import itertools
import random
import re

import yaml
import yaml.constructor

from collections import OrderedDict

class _OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError, exc:
                raise yaml.constructor.ConstructorError('while constructing a mapping',
                    node.start_mark, 'found unacceptable key (%s)' % exc, key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

class _YamlMixin:

    def to_yaml(self):
        return _yaml_ordered_dump(self)

    def to_base64(self):
        return self.to_yaml().encode('zlib_codec').encode('base64_codec')

def get_default_condition_filename(argv):
    import roslib.packages
    fn = os.path.basename(argv[0])
    fn,ext = os.path.splitext(fn)
    return os.path.join(roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                        'data','conditions',
                        '%s.yaml' % fn)

class ConditionCompat(OrderedDict):

    #make sure any named groups are named identically to the expected
    #fields in the experiment condition yaml file

    #https://regex101.com/r/tN5mQ4/1
    ROTATION_RE = re.compile("^(?P<cylinder_image>\w+\.png)/(?P<svg_path>\w+\.svg)/(?P<gain>[\d.+]+)/(?P<radius_when_locked>[\d.+-]+)/(?P<advance_threshold>[\d.+-]+)/(?P<z_gain>[\d.+]+)/?(?P<z_target>[\d.+-]+)?$")

    CONFLICT_RE = re.compile("^(?P<cylinder_image>\w+\.png)/(?P<svg_path>\w+\.svg)(?:/[\d.+-]+){1,5}/(?P<model_descriptor>\w+\.osg(?:\|[\d.+-]+)+)$")

    PERTURB_RE = re.compile("^\w+\.png\/\w+\.svg(?:\/[\d.+-]+){3,5}\/\w+\|.*$")

    CONFINE_RE = re.compile("^(?P<stimulus_filename>[\w.]+\.osg)/(?P<x0>[\d.+-]+)/(?P<y0>[\d.+-]+)/(?P<lag>[\d.+-]+)$")

    #https://regex101.com/r/yW6lG5/1
    TRANSLATION_RE = re.compile("^(?P<svg_path>\w+\.svg)/(?P<gain>[\d.+]+)/(?P<advance_threshold>[\d.+-]+)/(?P<z_gain>[\d.+]+)/(?P<star_size>[\d.+]+)/(?P<z_target>[\d.+-]+)$")

    def __init__(self, slash_string):
        OrderedDict.__init__(self)
        self._s = slash_string
        self._fake_names = []

        #translation experiments look like this
        # infinity07.svg/5.0/0.1/5.0/20.0/0.12
        match = ConditionCompat.TRANSLATION_RE.match(self._s)
        if match:
            self._fake_names.append('translation')
            self.update(match.groupdict())

        #rotation experiments look like this
        # checkerboard16.png/infinity.svg/0.3/-10.0/0.1/0.2
        # gray.png/infinity07.svg/0.3/-5.0/0.1/0.18/0.2
        # checkerboard16.png/infinity07.svg/0.3/-5.0/0.1/0.18/0.2
        match = ConditionCompat.ROTATION_RE.match(self._s)
        if match:
            self._fake_names.append('rotation')
            self.update(match.groupdict())

        #conflict experiments look like this
        # checkerboard16.png/infinity07.svg/0.3/-5.0/0.1/0.18/0.2/justpost1.osg|-0.1|-0.1|0.0
        match = ConditionCompat.CONFLICT_RE.match(self._s)
        if match:
            self._fake_names.append('conflict')
            self.update(match.groupdict())

        #perturb experiments look like this
        # checkerboard16.png/infinity.svg/0.3/-10.0/0.1/0.2/multitone_rotation_rate|rudinshapiro2|1.8|3|1|5||0.4|0.46|0.56|0.96|1.0|0.0|0.06
        # checkerboard16.png/infinity.svg/0.3/-10.0/0.1/0.2/step_rotation_rate|1.8|3|0.4|0.46|0.56|0.96|1.0|0.0|0.06
        match = ConditionCompat.PERTURB_RE.match(self._s)
        if match:
            #let the validation logic in sfe.perturb do the heavy work here.
            #lazy import for Santi
            from .perturb import is_perturb_condition_string
            if is_perturb_condition_string(self._s):
                self._fake_names.append('perturbation')
                self.update(match.groupdict())

        match = ConditionCompat.CONFINE_RE.match(self._s)
        if match:
            self._fake_names.append('confine')
            self.update(match.groupdict())

    def __repr__(self):
        return "<ConditionCompat '%s' (is %s)>" % (self._s, ','.join(self._fake_names))

    def is_type(self, *names):
        return any(name in self._fake_names for name in names)

    def to_slash_separated(self):
        return self._s

class Condition(OrderedDict, _YamlMixin):

    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        self.name = ''

    def __repr__(self):
        return "<Condition %s (%s)>" % (self.name, ','.join(self.keys()))

    @staticmethod
    def from_base64(txt,name=''):
        txt = txt.decode('base64_codec').decode('zlib_codec')
        c = Condition()
        c.update(yaml.load(txt, _OrderedDictYAMLLoader))
        c.name = name
        return c

    def to_slash_separated(self):
        return "/".join(str(v) for v in self.values())

    #these could use schema information in future, but for now just
    #enforce a naming scheme
    def is_type(self, *names):
        return any(name in self.name for name in names)


class Conditions(OrderedDict, _YamlMixin):

    def __init__(self, text_or_file_like, switch_order='seq', rng_seed=42):

        OrderedDict.__init__(self)

        try:
            txt = text_or_file_like.read()
        except AttributeError:
            txt = text_or_file_like

        VALID_SWITCH_STRATEGIES = {'seq', 'randstart', 'fullrand'}
        if switch_order not in VALID_SWITCH_STRATEGIES:
            raise ValueError('switch order must be one of %r, not "%s"' %
                             (VALID_SWITCH_STRATEGIES, switch_order))

        d = yaml.load(txt, _OrderedDictYAMLLoader)

        # provenance
        self.rng_seed = rng_seed
        self.switch_order = switch_order
        keys = d.keys()
        if self.switch_order != 'seq':
            self.rng = random.Random(rng_seed if rng_seed >= 0 else None)
            if self.switch_order == 'randstart':
                self.rng.shuffle(keys)
        else:
            self.rng = None

        for k in keys:
            v = d[k]
            if k.startswith('_') or k == 'uuid':
                continue

            c = Condition(v)
            c.name = k

            if c in self.values():
                raise ValueError("duplicate condition %s" % k)

            self[k] = c

    @staticmethod
    def from_base64(txt):
        return Conditions(txt.decode('base64_codec').decode('zlib_codec'))

    def next_condition(self, last_condition):
        if self.switch_order == 'fullrand':
            return self[self.rng.choice(self.keys())]
        else:
            d = itertools.cycle(self)
            for c in d:
                if last_condition is None:
                    return self[c]
                if self[c] == last_condition:
                    return self[d.next()]

    def first_condition(self):
        return self[self.keys()[0]]


def _yaml_ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _yaml_represent_ordereddict(dumper, data):
        value = []

        for item_key, item_value in data.items():
            node_key = dumper.represent_data(item_key)
            node_value = dumper.represent_data(item_value)

            value.append((node_key, node_value))

        return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)

    OrderedDumper.add_representer(OrderedDict, _yaml_represent_ordereddict)
    OrderedDumper.add_representer(Condition, _yaml_represent_ordereddict)
    OrderedDumper.add_representer(Conditions, _yaml_represent_ordereddict)

    return yaml.dump(data, stream, OrderedDumper, **kwds)

