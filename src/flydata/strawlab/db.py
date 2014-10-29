# coding=utf-8
"""Freeflight database utilities.

See also autodata.model.SQLModel
"""
from operator import itemgetter
from flydata.misc import run_db_statement

# Needed for historical reasons, until we get one single table
FREEFLIGHT_KNOWN_ARENAS = (('freeflight', 'flycave'),
                           ('flycube', 'flycube1'),
                           ('flycube', 'flycube2'),
                           ('flycube', 'flycube3'),
                           # ('flycube', 'flycube4'),
                           ('flycube', 'flycube5'),
                           ('flycube', 'flycube6'),
                           # ('flycube', 'flycube7'),
                           ('flycube', 'flycube8'),
                           ('flycube', 'flycube9'),
                           ('flycube', 'flycube10'),
                           ('flycube', 'flycube11'),
                           ('fishvr', 'fishtrax'))


def experiments_table(arena='fishtrax'):
    """Returns the name of the experiments table for a concrete arena."""
    return '/%s/experiment' % arena


def strawlab_freeflightdb_uri(protocol='postgresql',
                              url='flycave-postgres-server.strawlab.org',  # FIXME: get this from app config
                              dbname='flycube',
                              userid='autodata',
                              password=None):
    """Returns the URI for a database, made easy for strawlab."""
    # Auth info for our DB
    HARDCODED_IDS = {
        'resultswebserver': 'q69weld',   # Usually read only
        'autodata': 'wiePh1ad',          # Usually write, but not admin
    }
    if password is None:
        password = HARDCODED_IDS.get(userid, None)
    if password is None:
        raise Exception('Passwordless access not supported.')
    return '%s://%s:%s@%s/%s' % (protocol, userid, password, url, dbname)


def find_experiment(uuid, arenas=FREEFLIGHT_KNOWN_ARENAS):
    """Returns a tuple (database, arena, metadata_dictionary).

    N.B. the return type will change when we get rid of the multiple databases in our infrastructure.
    """
    for db, arena in arenas:
        uri = strawlab_freeflightdb_uri(dbname=db)
        table_name = experiments_table(arena)
        result = run_db_statement(uri, 'SELECT * FROM "%s" WHERE uuid=\'%s\'' % (table_name, uuid)).fetchone()
        if result is not None:
            return db, arena, dict(result.items())
    return None, None, None


def uuids_in_arena(db='fishvr', arena='fishtrax'):
    """Returns a list of uuids in the arena."""
    uri = strawlab_freeflightdb_uri(dbname=db)
    table_name = experiments_table(arena)
    return map(itemgetter(0), run_db_statement(uri, 'SELECT uuid FROM "%s"' % table_name).fetchall())