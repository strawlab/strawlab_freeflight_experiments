# coding=utf-8
"""Prints insert sql statements for the freeflight experiments table.
   Example with a couple of experiments that were missing from the DB due to infrastructure problems.
"""
from datetime import datetime
import pytz


def timestamp(year=2013, month=7, day=17, hour=9, minute=52, tzinfo=None):
    """Returns the UNIX timestamp for a concrete date and time.
    If tzinfo is None, we assume 'Europe/Vienna'.
    """
    if tzinfo is None:
        tzinfo = pytz.timezone('Europe/Vienna')
    dt = datetime(year=year, month=month, day=day, hour=hour, minute=minute, tzinfo=tzinfo)
    print int(dt.strftime("%s"))  # Also works
    return (dt - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()


def insert_statement(timestamp, uuid, title, description, genotype, age):
    return 'INSERT INTO "/fishtrax/experiment"(_ros_sql_timestamp_secs, start_secs,\n' \
           '                                   uuid, title, description, genotype, age)\n' \
           'VALUES (%d, %d, \'%s\', \'%s\', \'%s\', \'%s\', %d);' % \
           (timestamp, timestamp, uuid, title, description, genotype, age)


if __name__ == '__main__':

    print insert_statement(timestamp(year=2014, month=7, day=21, hour=13, minute=05),
                           'dd8c91f4117711e4906960a44c2451e5',
                           'translation',
                           '2 Medaka, same settings as bevore',
                           'Medaka AB',
                           20)
