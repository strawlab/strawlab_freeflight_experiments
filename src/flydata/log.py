# coding=utf-8
"""Common logger for the module."""
# TODO: conciliate with John's logging
import logging

_logger = logging.getLogger('flydata')  # 'discrimination_of_free_flies'
_logger.setLevel(logging.DEBUG)
debug = _logger.debug
info = _logger.info
warning = _logger.warning
error = _logger.error
logging.basicConfig()
