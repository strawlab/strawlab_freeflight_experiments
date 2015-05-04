INVALID_VALUE = 0xDEADBEEF

COLUMN_NAMES_TO_HUMAN_NAMES = {
    'rotation_rate':'Stimulus angular velocity',
    'dtheta':'Fly angular velocity',
}

COLUMN_NAMES_TO_UNITS = {
    'rotation_rate':'rad/s',
    'dtheta':'rad/s',
}

def get_human_names(col_name):
    return COLUMN_NAMES_TO_HUMAN_NAMES.get(col_name,col_name), COLUMN_NAMES_TO_UNITS.get(col_name,'')
