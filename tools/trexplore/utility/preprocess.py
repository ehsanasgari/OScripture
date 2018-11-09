

def preprocess_crs(verse):
    '''
    :param verse:
    :return: ti'n ==> ti 'n
    '''
    return verse.replace("'", " '").replace('  ', ' ')
