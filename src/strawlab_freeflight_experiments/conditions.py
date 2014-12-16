import os.path
import itertools

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

class Condition(OrderedDict, _YamlMixin):

    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        self.name = ''

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

    def __init__(self, text_or_file_like):
        try:
            txt = text_or_file_like.read()
        except AttributeError:
            txt = text_or_file_like

        d = yaml.load(txt, _OrderedDictYAMLLoader)

        OrderedDict.__init__(self)
        for k,v in d.iteritems():
            c = Condition(v)
            c.name = k
            self[k] = c

    @staticmethod
    def from_base64(txt):
         return Conditions(txt.decode('base64_codec').decode('zlib_codec'))

    def next_condition(self, last_condition):
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
