# coding=utf-8
"""Single experiment metadata from data-storage to python world and back."""
from datetime import datetime
import json
import os.path as op
import pytz

from flydata.misc import ensure_dir, giveupthefunc
from flydata.strawlab.db import find_experiment


################################
# The class...
################################


class FreeflightExperimentMetadata(object):
    """Metadata (unique id, genotype etc.) from a single Freeflight experiment.

    Essentially a lazy (kind of) immutable dictionary.
    Allows hard disk caching, automatic retrieval of data from the DB, IDE autocompletion and documents some keys.

    Parameters
    ----------
    uuid : string
        The experiment identifier

    json_file_or_dit : path string or None, default None
        Associated file from/to the metadata will be cached in json format.
        If None, no caching will be used.
        If a directory, a json file named "metadata#uuid=blah" will be used.
        Warning: if a file is intended, the path should have extension '.json'
                 if a directory is intended, the path should not have extension '.json'

    populate : boolean, default False
        If true, the metadata will be populated immediatly from the json file or the database if no file is associated.

    dictionary : dict, default None
        If not None, values in this metadata object will be added (or overriden) from this dictionary.
        This happens after population from json/DB, so these values take precedence.
    """

    __slots__ = ['_metadata', '_json']

    def __init__(self, uuid, json_file_or_dir=None, populate=False, dictionary=None):

        super(FreeflightExperimentMetadata, self).__init__()

        # Cache coordinates
        self._json = json_file_or_dir

        # The metadata, in a separate dictionary
        self._metadata = {'uuid': uuid}

        # Greedily populate
        if populate:
            self.genotype()

        # Override the keys with the dictionary
        if dictionary is not None:
            self._populate_from_dict(dictionary)

    #####
    # I (and my IDE) like explicit fields when possible...
    #####

    def prop(self, name):
        """Returns the value of the property, raising a KeyError if it does not exist."""
        if name not in self._metadata:
            self._populate_from_json()
        return self._metadata.get(name)

    def _magicprop(self):
        """Returns a value from the dictionary (or None) lazily loading the data.
        We magically know the property by looking at the name of a function in the stack.
        N.B. Do not use, it turns out this is really slow...
        """
        return self.prop(giveupthefunc(2).__name__)

    def uuid(self):
        """Returns the unique identifier of the experiment (string)."""
        return self.prop('uuid')
        # return self._magicprop()

    def user(self):
        """Returns who run the experiment (string or None)."""
        return self.prop('user')

    def hidden(self):
        """Returns whether the experiment has "hidden" status or not (boolean or None)."""
        return self.prop('hidden')

    def tags(self):
        """Returns the tags associated with the experiment as a comma-separated string (string or None)."""
        return self.prop('tags')

    def title(self):
        """Returns the title of the experiment (string or None)."""
        return self.prop('title')

    def description(self):
        """Returns the description of the experiment (string or None)."""
        return self.prop('description')

    def genotype(self):
        """Returns the title of the experiment (string or None)."""
        return self.prop('genotype')

    def age(self):
        """Returns the age of the flies (int or None)."""
        return self.prop('age')

    def arena(self):
        """Returns the arena (e.g. flycube2) in which the experiment was carried (string or None)."""
        return self.prop('arena')

    def num_flies(self):
        """Returns the number of flies in the experiment."""
        raise NotImplementedError('Things like number of flies, raising conditions and the like '
                                  'should also be accessible')

    def start_secs(self):
        """Returns the unix timestamp for when (UTC time) the experiment started (long or None)."""
        return self.prop('start_secs')

    def start_nsecs(self):
        """Returns the nanosec part of the unix timestamp for when (UTC time) the experiment started (long or None)."""
        return self.prop('start_nsecs')

    def start_t(self):
        """Returns the unix timestamp for when (UTC time) the experiment started (double or None)."""
        return self.start_secs() + self.start_nsecs() * 1E-9

    def start_asdatetime(self, tzinfo=pytz.timezone('Europe/Vienna')):
        return datetime.fromtimestamp(self.start_t(), tz=tzinfo)

    def stop_secs(self):
        """Returns the unix timestamp for when (UTC time) the experiment should approximatelly stop (long or None)."""
        return self.prop('stop_secs')

    def stop_nsecs(self):
        """Returns the nanosec part of the unix timestamp for when (UTC time)
        the experiment should approximatelly stop (long or None)."""
        return self.prop('stop_nsecs')

    def stop_t(self):
        """Returns the unix timestamp for when (UTC time) the experiment should approximatelly stop (double or None)."""
        return self.stop_secs() + self.stop_nsecs() * 1E-9

    def stop_asasdatetime(self, tzinfo=pytz.timezone('Europe/Vienna')):
        return datetime.fromtimestamp(self.stop_t(), tz=tzinfo)

    ######
    # Official mutability
    ######

    def apply_transform_inplace(self, **kwargs):
        """Applies single-attribute transformations to this metadata.

        This method endulges responsible API users with a way to break immutability by "official ways".

        Parameters
        ----------
        kwargs: pairs attribute_name -> transform
            - Each attribute name should correspond to the name of an attribute in the metadata
              (if it does not exist, a new attribute is created)
            - transform is a function that takes a metadata object and returns a single value;
              such value substitutes the attribute's previous value

        Examples
        --------
        If "md" is a metadata object, then this call will redefine the "genotype" and "stop_secs" attributes:
          md.apply_transform_inplace(
              genotype=lambda md: md.genotype() + '-normalized_genotype',
              stops_secs=lambda md: 10000000
          )
        """
        for key, fix in kwargs.iteritems():
            self._metadata[key] = fix(self)  # Awfully overcomplex

    ######
    # Retrieval and persistence
    ######

    def is_populated(self):
        """Have we already populated this metadata object?"""
        return 'genotype' in self._metadata  # Kinda lame

    def _populate_from_dict(self, dictionary):
        """Adds or overrides the (key, value) pairs with all the pairs from "dictionary"."""
        if dictionary.get('uuid', None) != self.uuid():
            raise Exception('The experiment UUID must be consistent with the UUID in the dictionary')
        for k, v in dictionary.iteritems():
            self._metadata[k] = v

    def _populate_from_db(self):
        """Lazily Populates the dictionary from the freeflight DB."""

        # Laziness (careful with infinite recursion if changed)
        if self.is_populated():
            return

        db, arena, data = find_experiment(self.uuid())
        if data is None:
            raise Exception('Cannot find experiment with UUID %s' % self.uuid())
        # N.B. more sanity checks, like uniqueness, should not be needed if we properly design the DB

        # This will be not needed with the new DB schema on
        if 'arena' not in data:
            data['arena'] = arena

        # Remove ros-sql cruft and DB ids
        data.pop('_ros_sql_timestamp_secs', None)
        data.pop('_ros_sql_timestamp_nsecs', None)
        data.pop('id', None)

        self._populate_from_dict(data)

    def json_path(self):
        """Returns the path to the default json file to which this metadata will be cached."""
        if self._json is None:            # Caching is disabled
            return None
        if self._json.endswith('.json'):  # Is this a json file?
            return self._json
        ensure_dir(self._json)  # Assume a directory, generate default name
        path = op.join(self._json, 'metadata#uuid=%s.json' % self.uuid())
        return path

    def _populate_from_json(self):
        """Lazily populate this metadata object with information from the json file,
        falling back to the DB if the json file does not exist.
        """
        if self.is_populated():  # Already populated (lame test)
            return
        if self.json_path() is None:     # We actually do not cache in disk, db fallback
            self._populate_from_db()
        elif not op.isfile(self.json_path()):  # The cache is not there yet, db query + create cache json
            self._populate_from_db()
            self.to_json_file()
        else:                                   # Populate from an already existent cache
            with open(self.json_path()) as reader:
                my_dict = json.load(reader)
                if not self.uuid() == my_dict.get('uuid', None):
                    raise Exception('The UUIDs do not coincide (%s != %s)' % (self.uuid, my_dict.get('uuid', None)))
                self._populate_from_dict(my_dict)

    ######
    # Flatten
    ######

    def flatten(self):
        """Returns a list of tuples (key, value) from this metadata object, sorted by key."""
        return sorted(self._metadata.items())

    ######
    # To/from json
    ######

    def to_json_string(self):
        return json.dumps(self._metadata, indent=2, ensure_ascii=False).encode('utf8')

    @classmethod
    def from_json_string(cls, json_string, json_file_or_dir=None):
        dictionary = json.loads(json_string)
        uuid = dictionary.get('uuid', None)
        if uuid is None:
            if json_file_or_dir is None:
                raise Exception('The json string does not seem to have any uuid field')
            else:
                raise Exception('The json string (from %s) does not seem to have any uuid field' % json_file_or_dir)
        return cls(uuid=uuid, json_file_or_dir=json_file_or_dir)

    def to_json_file(self, json_path=None, overwrite=False):
        if json_path is None:
            json_path = self.json_path()
        if json_path is not None:
            if overwrite or not op.isfile(json_path):
                with open(json_path, 'w') as writer:
                    writer.write(self.to_json_string())

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as reader:
            return cls.from_json_string(reader.read(), json_file_or_dir=json_file)

    ######
    # From message
    ######
    @classmethod
    def from_message(cls, message):
        """Loads from a ros experiment message."""
        #
        # Easy to do by hand (or look at bag2hdf5)
        # Does ros help us here somehow?
        #
        # A good function to convert generic ros messages,
        # including nested types, to python values, should prove useful.
        # It could be (re)used here, in bag2hdf5, ros_sql...
        #
        # If doing by hand, this is the current def in flycave:
        # ------------------
        # time start  <- recall we get secs and nanosecs as int64 from ros,
        #                atm we store a float64 timestamp
        # string uuid
        # string title
        # string description
        # string genotype
        # int8 age
        # bool hidden
        # string user
        # string tags
        # time stop
        # ------------------
        #
        raise NotImplementedError()

    ######
    # To/from pandas
    ######

    @staticmethod
    def to_pandas(metadatas):
        """Generates a pandas dataframe from a list of metadata objects.
        Order of metadata objects will be respected (the first object will be the first row and so on).
        Duplicated experiments (as per uuid) are removed.

        Parameters
        ----------
        metadatas : iterable of FreeflightExperimentMetadata objects
            The metadata information that will go into the dataframe.

        Returns
        -------
        The pandas dataframe with a row per metadata.
        """
        from pandas import DataFrame
        return DataFrame({md.uuid(): md._metadata for md in metadatas}).T

    @staticmethod
    def from_pandas(df):
        """Returns a dictionary {uuid -> metadata} from a pandas dataframe obtained with "to_pandas" method.
        N.B. no validity checks are done.
        """
        def md_builder(uuid, the_dict):
            md = FreeflightExperimentMetadata(uuid, populate=False)
            md._populate_from_dict(the_dict)
            return md
        return {uuid: md_builder(uuid, the_dict) for uuid, the_dict in df.T.to_dict().iteritems()}

    ######
    # Magics
    ######

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '%s(uuid=%s, cache_dir=%r, populate=%r, dictionary=%r)' % \
               (self.__class__.__name__, self.uuid(), self._json, self.is_populated(), self._metadata)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._metadata == other._metadata

    def __ne__(self, other):
        return not self.__eq__(other)


################################
# Examples...
################################

if __name__ == '__main__':

    from itertools import chain

    # Some UUIDs from Katja
    PDF_GAL4xKP041 = (
        # Gal4 in PDF-producer-cells expressing the KP041 (reaper) gene leading to apoptosis
        '77023d5eb9b511e3b02210bf48d76973',  # (UTC 15:51) 17:51-10:00  PDf-GAL4 x KP041  2014-04-01 17:51:18
        '185607a2bc0f11e3a8d110bf48d76973',  # (UTC 15:37) 17:37-10:00  PDF-GAL4 x KP041  2014-04-04 17:37:57
        'ca9efe66d46d11e3b78110bf48d76973',  # 17:56-10:00  PDF-GAL4 x KP041 (f)  2014-05-05 17:56:16
        '6b7f3facd85c11e3a99310bf48d76973',  # 18:02-10:00  PDF-GAL4 xKP041       2014-05-10 18:02:00
    )

    PDF_GAL4xCS = (
        # Control 1: Gal4 in PDF-producer-cells combined with control strain
        '17ff102eba7e11e385af10bf48d76973',  # (15:47 UTC) 17:47-10.00  PDF-GAL4 x CS  2014-04-02 17:47:28
        'd3d1b92cbe6c11e3b54f10bf48d76973',  # (UTC 15:53) 17-53-10:00  PDF-GAL4 x CS  2014-04-07 17:53:57
        '8ce92230d9eb11e3ad5010bf48d76973',	 # 17:39-10:00  PDF-GAL4 x CS  2014-05-12 17:39:05
        '769e6cf8dab611e3b64f10bf48d76973',  # 17:51-10:00  PDF-GAL4 x CS  2014-05-13 17:51:36
    )

    KP041xCS = (
        # Control 2: KP041 combined with control strain
        '900bfc06bb4811e3a14f10bf48d76973',  # (UTC 15:56) 17:56-10:00  KP041 x CS  2014-04-03 17:56:47
        '127f617ebf3511e3a69810bf48d76973',  # (UTC 15:47) 17:47-10:00  KP041 x CS  2014-04-08 17:47:21
        '62fc1158d53611e3b09710bf48d76973',  # 17:52-10:00  KP041 x CS  2014-05-06 17:52:11
        '5827179ed92511e3b15910bf48d76973',  # 18:00-10:00  KP041 x CS  2014-05-11 18:00:16
    )

    # Damn long names
    FFEM = FreeflightExperimentMetadata

    # Time-zone
    tz = pytz.timezone('Europe/Vienna')

    # Load the metadata for each experiment
    metadatas = [FFEM(uuid) for uuid in chain(PDF_GAL4xKP041, PDF_GAL4xCS, KP041xCS)]

    # Accessing the different fields is easy
    for md in metadatas:
        print '\t'.join((md.uuid(),
                         md.genotype(),
                         md.arena(),
                         str(md.age()),
                         str(datetime.fromtimestamp(md.start_t(), tz=tz))))

    # To pandas...
    df = FFEM.to_pandas(metadatas)

    # Create a new column start from timestamp Viennoise datetime (FIXME should be provided by metadata)
    df['start'] = df.apply(lambda md: datetime.fromtimestamp(md['start_secs'] + 1E-9 * md['start_nsecs'], tz=tz),
                           axis=1)

    # Pandas style selection might come handy...
    print 'These were the experiments carried on May 2014...'
    in_may = (df['start'] > datetime(year=2014, month=4, day=30,
                                     hour=23, minute=59, second=59,
                                     tzinfo=pytz.timezone('Europe/Vienna'))) &\
             (df['start'] < datetime(year=2014, month=6, day=1, tzinfo=pytz.timezone('Europe/Vienna')))
    print df[in_may][['genotype', 'start']].sort(columns='start')

#
# Probably we would like to separate persistence from this class, although it is quite convenient to have it here
#
# If going for more functionality (e.g. sync/update the DB) it could repay to use an ORM
#
# Starting from UUIDs is ok but...
#   we should provide query capabilities here (ala autodata.SQLModel) or teach SQL to people
#
# Genotype string normalization and the like could come here
#