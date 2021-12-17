# Author: Paul Reiners

def get_id_to_region_mapping(mapping_file_name, separator=None):
    """
    Create a map from region ID to region name from a from a FreeSurfer-style look-up table.

    This functions parses a FreeSurfer-style look-up table.  It then returns a map that maps region IDs to their names.

    Parameters
    ----------
    mapping_file_name : string
        Name or path to the look-up table.
    separator : string
        String delimiter separating parts of lines in look-up table.

    Returns
    -------
    dict
        A map from the ID of a region to its name.
    """
    file = open(mapping_file_name, 'r')
    lines = file.readlines()

    id_to_region = {}
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        if separator:
            parts = line.split(separator)
        else:
            parts = line.split()
        region_id = int(parts[0])
        region = parts[1]
        id_to_region[region_id] = region
    return id_to_region
