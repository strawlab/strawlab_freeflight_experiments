# coding=utf-8
"""Prints insert sql statements for the freeflight experiments table.
   Example with a couple of experiments that were missing from the DB due to infrastructure problems.
"""
from datetime import datetime
import pytz


def timestamp(year=2014, month=2, day=21, hour=17, minute=45, tzinfo=None):
    """Returns the UNIX timestamp for a concrete date and time.
    If tzinfo is None, we assume 'Europe/Vienna'.
    """
    if tzinfo is None:
        tzinfo = pytz.timezone('Europe/Vienna')
    dt = datetime(year=year, month=month, day=day, hour=hour, minute=minute, tzinfo=tzinfo)
    # print int(date.strftime("%s"))  # Also works
    return (dt - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()


def insert_statement(timestamp, uuid, title, description, genotype, age):
    return 'INSERT INTO "/flycube1/experiment"(_ros_sql_timestamp_secs, start_secs,\n' \
           '                                   uuid, title, description, genotype, age)\n' \
           'VALUES (%d, %d, \'%s\', \'%s\', \'%s\', \'%s\', %d);' % \
           (timestamp, timestamp, uuid, title, description, genotype, age)


if __name__ == '__main__':

    print insert_statement(timestamp(year=2014, month=2, day=21, hour=17, minute=45),
                           '76de77789b1711e3b55c10bf48d76973',
                           'rotation',
                           '7 female flies, 4 days old, 17:45-11:00',
                           'Kir2.1 x PDF-GAL4',
                           4)

    print insert_statement(timestamp(year=2014, month=2, day=23, hour=17, minute=51),
                           'c92120a09caa11e3b1d810bf48d76973',
                           'rotation',
                           '4 female flies, 4 days old, 17:51-11:00',
                           'Dcr2-nsyb x RNAi_PDF',
                           4)