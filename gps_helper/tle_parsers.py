def get_celestrak_sv(tle):
    """
    Place a text file of Celestrak GPS TLEs into a dictionary with
    keys of the form 'PRN xx' for the SV of interest

    :param tle: Path to two line element set file.
    :return: dict
    """
    GPS_sv_dict = {}
    with open(tle, 'rt') as f:
        for line in f:
            if line[0:3] == 'GPS':
                PRN = line[line.find('PRN'):line.find(')')]
            elif line[0] == '1':
                tle_ln1 = line.strip('\n')
            elif line[0] == '2':
                tle_ln2 = line.strip('\n')
                GPS_sv_dict.update({PRN: (tle_ln1, tle_ln2)})
            else:
                UserWarning('File structure incorrect')
                break
    return GPS_sv_dict


def get_spacetrack_sv(tle):
    """
    Place a text file of Spacetrack GPS TLEs into a dictionary with keys of the form 'PRN xx' for the SV of interest

    :param tle: Path to two line element set file
    :return: dict
    """
    gps_sv_dict = {}
    ns = 'NAVSTAR'
    with open(tle, 'rt') as tf:
        for line in tf:
            if ns in line:
                prn = line[line.find(ns):line.find('(')-1]
                prn = prn.replace(ns, 'PRN')
            elif line[0] == '1':
                tle_ln1 = line.strip('\n')
            elif line[0] == '2':
                tle_ln2 = line.strip('\n')
                gps_sv_dict.update({prn: (tle_ln1, tle_ln2)})
            else:
                UserWarning('File structure incorrect')
                break
    return gps_sv_dict