# coding=utf-8
import json
from flydata.misc import can_connect
from flydata.strawlab.metadata import FreeflightExperimentMetadata, strawlab_freeflightdb_uri


def testFreeflightMetadata(tmpdir):

    # These tests require access to the strawlab freeflight DB
    if not can_connect(strawlab_freeflightdb_uri())[0]:
        assert 'WARNING: we have no access to the stralab database, skipping metadata tests' is None
        return

    #
    # We compare to experiment 2c8bfb5a8c2b11e3950210bf48d76973
    # ---------------------------------------
    # SELECT *
    # FROM "/flycube1/experiment"
    # WHERE uuid='2c8bfb5a8c2b11e3950210bf48d76973';
    # ---------------------------------------
    #

    #
    # Populate from the database
    #
    md = FreeflightExperimentMetadata(uuid='2c8bfb5a8c2b11e3950210bf48d76973')
    assert md.uuid() == u'2c8bfb5a8c2b11e3950210bf48d76973'
    assert md.title() == u'rotation'
    assert md.description() == u'7 female flies, 4 days old, temp. aircond. 22 -->23°C, 18-11h'
    assert md.genotype() == u'TNT active x PDF-GAL4'
    assert md.age() == 4
    assert md.start_secs() == 1391360281
    assert md.start_nsecs() == 10214090
    assert md.stop_secs() is None
    assert md.stop_nsecs() is None
    assert md.hidden() is False
    assert md.user() is None
    assert md.tags() is None

    #
    # JSON Caching - writing
    #
    cache_dir = tmpdir.mkdir('cache')
    expected_cache_json = cache_dir.join('metadata#uuid=2c8bfb5a8c2b11e3950210bf48d76973.json')
    md = FreeflightExperimentMetadata(uuid='2c8bfb5a8c2b11e3950210bf48d76973',
                                      json_file_or_dir=str(cache_dir),
                                      populate=True)
    assert expected_cache_json.check(file=1)
    assert md.uuid() == u'2c8bfb5a8c2b11e3950210bf48d76973'
    assert md.title() == u'rotation'
    assert md.description() == u'7 female flies, 4 days old, temp. aircond. 22 -->23°C, 18-11h'
    assert md.genotype() == u'TNT active x PDF-GAL4'
    assert md.age() == 4
    assert md.start_secs() == 1391360281
    assert md.start_nsecs() == 10214090
    assert md.stop_secs() is None
    assert md.stop_nsecs() is None
    assert md.hidden() is False
    assert md.user() is None
    assert md.tags() is None

    #
    # Populate from dictionary
    #
    with expected_cache_json.open() as reader:
        dictionary = json.load(reader)
    dictionary['user'] = 'tester'
    md = FreeflightExperimentMetadata(uuid='2c8bfb5a8c2b11e3950210bf48d76973',
                                      json_file_or_dir=str(cache_dir),
                                      populate=True,
                                      dictionary=dictionary)
    assert md.uuid() == u'2c8bfb5a8c2b11e3950210bf48d76973'
    assert md.title() == u'rotation'
    assert md.description() == u'7 female flies, 4 days old, temp. aircond. 22 -->23°C, 18-11h'
    assert md.genotype() == u'TNT active x PDF-GAL4'
    assert md.age() == 4
    assert md.start_secs() == 1391360281
    assert md.start_nsecs() == 10214090
    assert md.stop_secs() is None
    assert md.stop_nsecs() is None
    assert md.hidden() is False
    assert md.user() == 'tester'
    assert md.tags() is None

    #
    # JSON Caching - reading
    #
    with expected_cache_json.open('w') as writer:
        json.dump(dictionary, writer, indent=2)
    md = FreeflightExperimentMetadata(uuid='2c8bfb5a8c2b11e3950210bf48d76973',
                                      json_file_or_dir=str(cache_dir),
                                      populate=False)
    assert md.uuid() == u'2c8bfb5a8c2b11e3950210bf48d76973'
    assert md.title() == u'rotation'
    assert md.description() == u'7 female flies, 4 days old, temp. aircond. 22 -->23°C, 18-11h'
    assert md.genotype() == u'TNT active x PDF-GAL4'
    assert md.age() == 4
    assert md.start_secs() == 1391360281
    assert md.start_nsecs() == 10214090
    assert md.stop_secs() is None
    assert md.stop_nsecs() is None
    assert md.hidden() is False
    assert md.user() == 'tester'
    assert md.tags() is None

    #
    # Flatten
    #
    flattened = md.flatten()
    assert flattened == [(u'age', 4),
                         (u'arena', u'flycube1'),
                         (u'description', u'7 female flies, 4 days old, temp. aircond. 22 -->23°C, 18-11h'),
                         (u'genotype', u'TNT active x PDF-GAL4'),
                         (u'hidden', False),
                         (u'start_nsecs', 10214090),
                         (u'start_secs', 1391360281),
                         (u'stop_nsecs', None),
                         (u'stop_secs', None),
                         (u'tags', None),
                         (u'title', u'rotation'),
                         (u'user', u'tester'),
                         ('uuid', u'2c8bfb5a8c2b11e3950210bf48d76973')]

    #
    # To/From pandas and magics
    #
    df = FreeflightExperimentMetadata.to_pandas([md] * 5)
    assert len(df) == 1, 'Duplicates should be detected and removed'
    assert df.loc[u'2c8bfb5a8c2b11e3950210bf48d76973'][u'hidden'] is False
    assert df.loc[u'2c8bfb5a8c2b11e3950210bf48d76973'][u'user'] == u'tester'

    md_roundtrip = FreeflightExperimentMetadata.from_pandas(df)
    assert md == md_roundtrip[u'2c8bfb5a8c2b11e3950210bf48d76973']
    assert not md != md_roundtrip[u'2c8bfb5a8c2b11e3950210bf48d76973']

    # Cleanup - tmpdir should be a fixture...
    tmpdir.remove()