import os
import warnings
import numpy as np
from sgp4.io import twoline2rv
from sgp4.io import jday
from sgp4.earth_gravity import wgs84
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    warnings.warn("matplotlib not imported due to import error")
try:
    from mayavi import mlab
    from mayavi.sources.builtin_surface import BuiltinSurface
except ImportError:
    warnings.warn("mayavi not imported due to import error")

radius = 6371669.9


class GPSDataSource(object):
    """
    Obtain GPS SV position in ECEF via TLEs and SGP4 orbit propagation.
    Additionally integrate user trajectories in ENU about an input lla location.
    The resulting data sets

    Mark Wickert January 2018
    """

    def __init__(self, gps_tle_file, rx_sv_list=('PRN 03', 'PRN 09', 'PRN 22', 'PRN 26'),
                 ref_lla=(38.8454167, -104.7215556, 1903.0), ts=1.0):
        """
        Parameters
        ----------
        gps_tle_file : A text file extracted from Celestrak Web Site
        ref_lla : A 3 element tuple of lat(deg), lon(deg), and ele(m)

        Returns
        -------
        None : This is the constructor method

        Notes
        -----
        Default lla is CosmicAES COS office
        """
        self.ref_lla = ref_lla
        self.GPS_TLE_file = gps_tle_file
        self.Rx_sv_list = rx_sv_list
        """
        Perform coordinate conversion
        """
        # convert from llh (or lla) to earth centric fixed
        self.ref_ecef = self.llh2ecef(self.ref_lla)

        """
        Read GPS TLEs into dictionary
        """
        self.GPS_sv_dict = self.get_gps_sv(self.GPS_TLE_file)

        """
        Initialize Four SGP4 satellite objects in tuple satellite 
        """
        self.satellite = []
        for k in range(4):
            PRN = self.Rx_sv_list[k]
            self.satellite.append(twoline2rv(self.GPS_sv_dict[PRN][0], self.GPS_sv_dict[PRN][1], wgs84))

        """
        Time offset relative to satellite propagation start time
        """
        self.Ts = ts
        self.t_delta = np.array([0])
        self.N_sim_steps = 0

    def get_gps_sv(self, tle):
        """
        Place a text file of Celestrak GPS TLEs into a dictionary with
        keys of the form 'PRN xx' for the SV of interest

        :param tle: Two line element set as a file.
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
                    print('File structure incorrect')
                    break
        return GPS_sv_dict

    def create_sv_data_set(self, yr2, mon, day, hr, minute):
        """
        This method returns a numpy ndarrays containing target TDOA in seconds, FDOA in Hz,
        and the time span in minutes

        Parameters
        ----------
        yr2 : year as two digit integer, e.g., 2018 <=> 18)
        mon : month as two digit integer
        day : day as two digits, but can include fractional values
        hr : hour as two digits, but can include fractional values
        minute : minute as two digits, but can include fractional values
        sec : second as two digit integer, but can include fractional values; seconds default is 0

        Returns
        -------
        SV_Pos : A 3D ndarray containing the SV (x,y,x) position values (columns) for each of 4
                 satellites (rows) in meters corresponding to the times in t (slice)
        SV_Vel : A 3D ndarray containing the SV (x,y,x) velocity values (columns) for each of 4
                 satellites (rows) in meters corresponding to the times in t (slice)

        Notes
        -----


        Examples
        --------

        """
        "Fill position and velocity arrays"
        SV_Pos = np.zeros((4, 3, self.N_sim_steps))
        SV_Vel = np.zeros((4, 3, self.N_sim_steps))
        year = 2000 + yr2
        for n in range(4):
            for k, tk in enumerate(self.t_delta):
                SV_Pos[n, :, k], SV_Vel[n, :, k], gst = self.propagate_ecef(self.satellite[n], year, mon, day, hr,
                                                                            minute, tk)
        return SV_Pos, SV_Vel

    def user_traj_gen(self, route_list, vmph, yr2=18, mon=1, day=14, hr=20, minute=0):
        """
        GPS user trajectory generator

        Mark Wickert January 2018
        """
        # Miles per time step Ts
        Dmile = vmph * self.Ts / 3600
        Nsegs = len(route_list)
        Steps = []
        Steps_Total = 0
        kSteps = 0
        USER_Pos_enu_old = np.zeros(3)
        if vmph > 0:
            for k in range(Nsegs):
                Steps.append(int(np.floor(abs(route_list[k][1]) / vmph * 3600)))
                Steps_Total += Steps[k]
            USER_Pos_enu = np.zeros((Steps_Total, 3))  # Units of miles
            USER_Pos_ecef = np.zeros((Steps_Total, 3))  # Units of meters
            for n in range(Nsegs):
                for m in range(Steps[n]):
                    USER_Pos_enu[kSteps, :] = USER_Pos_enu_old
                    if route_list[n][0] == 'e':
                        USER_Pos_enu[kSteps, 0] += Dmile * np.sign(route_list[n][1])
                    elif route_list[n][0] == 'n':
                        USER_Pos_enu[kSteps, 1] += Dmile * np.sign(route_list[n][1])
                    elif route_list[n][0] == 'u':
                        USER_Pos_enu[kSteps, 2] += Dmile * np.sign(route_list[n][1])
                    else:
                        print("Route direction must be 'e', 'n', or 'u'")
                    USER_Pos_enu_old = USER_Pos_enu[kSteps, :]
                    USER_Pos_ecef[kSteps, :] = enu2ecef(USER_Pos_enu[kSteps, :] * 1609.344, self.ref_ecef,
                                                        self.ref_lla[0], self.ref_lla[1])
                    kSteps += 1
        else:
            Steps_Total = abs(vmph)
            USER_Pos_enu = np.zeros((Steps_Total, 3))  # Units of miles
            USER_Pos_ecef = np.zeros((Steps_Total, 3))  # Units of meters
            for m in range(Steps_Total):
                USER_Pos_ecef[m, :] = enu2ecef(USER_Pos_enu[m, :] * 1609.344, self.ref_ecef,
                                               self.ref_lla[0], self.ref_lla[1])
        self.t_delta = np.arange(0, Steps_Total) * self.Ts
        self.N_sim_steps = Steps_Total

        # Generate the corresponding SV positions and velocities
        SV_Pos, SV_Vel = self.create_sv_data_set(yr2, mon, day, hr, minute)
        return USER_Pos_enu, USER_Pos_ecef, SV_Pos, SV_Vel

    def propagate_ecef(self, satellite, year, mon, day, hr, minute, sec):
        """
        This method is responsible for propagating the satellite object, denoted satellite,
        to a specified GMT time. Coded to Python from the work of Charles Rino, (c) 2010,
        in MATLAB. See the function body details.

        Parameters
        ----------
        satellite : Input satellite object created by the sgp4 method twolinerv()
        year : year as four digit integer, e.g., 2013
        mon : month as two digit integer
        day : day as two digits, but can include fractional values
        hr : hour as two digits, but can include fractional values
        minute : minute as two digits, but can include fractional values
        sec : second as two digit integer, but can include fractional values

        Returns
        -------
        xsat_ecef : Satellite position in ECF coordinates
        vsat_ecef : Satellite velocity in ECF coordinates
        gst : Greenwich sidereal time

        Notes
        -----
        This is a private function used only by the class.
        When the class constructor is called, primary and secondary satellite objects are created
        using the input TLEs, e.g.,
        self.sat_primary = twoline2rv(self.tle_primary[0],
                                      self.tle_primary[1],wgs84)
        self.sat_secondary = twoline2rv(self.tle_secondary[0],self.tle_secondary[1],wgs84)
        The function propagate_ecf is used in all calculations where TDOA and FDOA are calculated.

        Examples
        --------
        >>> # As called from r_r_dot_ecf
        >>> position, velocity, gst =
                              self.propagate_ecef(sv,2000+yr2,
                                                 mon,day,hr,
                                                 minute,sec)

        """
        # USAGE: satrec, xsat_ecef, vsat_ecf, gst = sgp4_ecf(satrec,tsince);
        # Convert spg4 eci output to ecf
        # This segment converts eci coordinates to ecf
        #
        # Copyright (c) 2010, Charles Rino
        # All rights reserved.
        #
        # Redistribution and use in source and binary forms, with or without
        # modification, are permitted provided that the following conditions are
        # met:
        #
        # * Redistributions of source code must retain the above copyright
        # notice, this list of conditions and the following disclaimer.
        # * Redistributions in binary form must reproduce the above copyright
        # notice, this list of conditions and the following disclaimer in
        # the documentation and/or other materials provided with the distribution
        #
        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
        # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
        # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
        # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
        # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
        # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
        # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
        # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
        # POSSIBILITY OF SUCH DAMAGE.
        # Recoded to Python by Mark Wickert, July 2013

        xsat_eci, vsat_eci = satellite.propagate(year, mon, day, hr, minute, sec)
        " Compute Greenwich Apparent Sidereal Time"
        gst = self.gstime(jday(year, mon, day, hr, minute, sec))
        " Now rotate the coordinates"
        CGAST = np.cos(gst)
        SGAST = np.sin(gst)
        xsat_ecef = np.zeros(3)
        vsat_ecef = np.zeros(3)
        xsat_ecef[0] = xsat_eci[0] * CGAST + xsat_eci[1] * SGAST
        xsat_ecef[1] = -xsat_eci[0] * SGAST + xsat_eci[1] * CGAST
        xsat_ecef[2] = xsat_eci[2]
        "Apply rotation to convert velocity vector from ECI to ECEF coordinate"
        OMEGAE = 7.29211586E-5  # Earth rotation rate in rad/s
        vsat_ecef[0] = vsat_eci[0] * CGAST + vsat_eci[1] * SGAST + OMEGAE * xsat_ecef[1]
        vsat_ecef[1] = -vsat_eci[0] * SGAST + vsat_eci[1] * CGAST - OMEGAE * xsat_ecef[0]
        vsat_ecef[2] = vsat_eci[2]
        # Set units to m and m/s from km and km/s
        return xsat_ecef * 1e3, vsat_ecef * 1e3, gst

    def gstime(self, jdut1):
        """
        This method converts the Julian date to Greenwich sidereal time (gst)

        Parameters
        ----------
        jut1 : Julian date

        Returns
        -------
        gst : Greenwich sidereal time

        Notes
        -----
        A support function from the original David Vallado SGP4 recoded to Python. This is a
        private method.

        Examples
        --------

        >>> # Inside propagate_ecf() make this call:
        >>> gst = self.gstime(jday(year,mon,day,hr,minute,sec));

        """
        # -----------------------------------------------------------------------------
        #
        #                            function gstime
        #
        #   this function finds the greenwich sidereal time (iau-82).
        #
        #   author        : david vallado                  719-573-2600    7 jun 2002
        #
        #   revisions
        #                 -
        #
        #   inputs          description                    range / units
        #     jdut1       - julian date of ut1             days from 4713 bc
        #
        #   outputs       :
        #     gst         - greenwich sidereal time        0 to 2pi rad
        #
        #   locals        :
        #     temp        - temporary variable for reals   rad
        #     tut1        - julian centuries from the
        #                   jan 1, 2000 12 h epoch (ut1)
        #
        #   coupling      :
        #
        #  references    :
        #    vallado       2007, 193, Eq 3-43
        #
        # -----------------------------------------------------------------------------
        # Recoded to Python by Mark Wickert, July 2013

        twopi = 2.0 * np.pi
        deg2rad = np.pi / 180.0
        # ------------------------  implementation   ------------------
        tut1 = (jdut1 - 2451545.0) / 36525.0
        temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1
        temp += (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841
        # 360/86400 = 1/240, to deg, to rad
        temp = np.mod(temp * deg2rad / 240.0, twopi)
        # ------------------------ check quadrants --------------------
        if temp < 0.0:
            temp += twopi
        gst = temp
        return gst

    def earth_model(self):
        """
        Define the constants from the WGS-84 ellipsoidal Earth model.


        Parameters
        ----------
        None

        Returns
        -------
        a : semi-major axis of the Earth ellipsoid model
        f : flattening

        Notes
        -----
        The World Geodetic System (WGS) is a standard for use in cartography, geodesy, and navigation.
        The latest revision is WGS-84.

        """
        a = 6378137.0  # meters
        f = 1.0 / 298.257223563
        return a, f

    def llh2ecef(self, llh):
        """
        Convert lat,lon,hgt geographic coords to X,Y,Z Earth Centered Earth
        Fixed (ecef) or just (ecf) coords.

        Parameters
        ----------
        llh : A three element ndarray containing latitude(lat), longitude (lon), and altitude (a) or height (hgt), all in meters

        Returns
        -------
        x : The ecef x coordinate
        y : The ecef y coordinate
        z : The ecef z coordinate

        Notes
        -----
        This is a private function that computes:
        N = a/sqrt( 1 - f*(2-f)*sin(lat)*sin(lat) )
        X = (N + h)*cos(lat)*cos(lon)
        Y = (N + h)*cos(lat)*sin(lon)
        Z = ((1-f)^2 * N + h)*sin(lat)
        by also calling EarthModel()

        Examples
        --------

        """

        lat = llh[0] * np.pi / 180.
        lon = llh[1] * np.pi / 180.
        hgt = llh[2]

        ecf = np.zeros(3)
        " Set up WGS-84 constants."
        a, f = self.earth_model()

        " Store some commonly used values."
        slat = np.sin(lat)
        N = a / np.sqrt(1 - f * (2 - f) * slat ** 2)
        Nplushgtclat = (N + hgt) * np.cos(lat)

        x = Nplushgtclat * np.cos(lon)
        y = Nplushgtclat * np.sin(lon)
        z = ((1 - f) ** 2 * N + hgt) * slat

        return np.array([x, y, z])

    def days2mdh(self, year, days):
        """
        This function converts the day of the year, days, to the equivalent month day,
        hour, minute and second. From Vallado's original code translated to Python.

        Parameters
        ----------
        year : The year as a four digit number, e.g., 2013
        days : day of the year as a decimal number

        Returns
        -------
        mo : month as two digit integer
        day : day as two digits
        hour : hour as two digits
        minute : minute as two digits,
        sec : second as two digit integer, but can include fractional values

        Notes
        -----
        A support function in the class, currently not utilized by any methods.

        """
        # ---------------------------------------------------------------------
        #
        #                           function days2mdh
        #
        #  this function converts the day of the year, days, to the equivalent
        #  month day, hour, minute and second.
        #
        #  author        : david vallado            719-573-2600   22 jun 2002
        #
        #  revisions
        #                -
        #
        #  inputs          description              range / units
        #    year        - year                     900 .. 2100
        #    days        - julian day of the year   0.0  .. 366.0
        #
        #  outputs       :
        #    mon         - month                    1 .. 12
        #    day         - day                      1 .. 28,29,30,31
        #    hr          - hour                     0 .. 23
        #    minute      - minute                   0 .. 59
        #    sec         - second                   0.0 .. 59.999
        #
        #  locals        :
        #    dayofyr     - day of year
        #    temp        - temporary extended values
        #    inttemp     - temporary integer value
        #    i           - index
        #    lmonth(12)  - integer array containing the number of days per month
        #
        #  coupling      :
        #    none.
        #
        # [mon,day,hr,minute,sec] = days2mdh ( year,days);
        # ---------------------------------------------------------------------
        # Recoded to Python by Mark Wickert, July 2013

        lmonth = np.zeros(12)
        " --------------- set up array of days in month  --------------"
        for i in range(12):
            lmonth[i] = 31
            if i + 1 == 2:
                lmonth[i] = 28
            if i + 1 == 4 or i + 1 == 6 or i + 1 == 9 or i + 1 == 11:
                lmonth[i] = 30
        dayofyr = np.floor(days)
        " ----------------- find month and day of month ---------------"
        if np.mod(year - 1900, 4) == 0:
            lmonth[2 - 1] = 29
        i = 1 - 1
        inttemp = 0
        while (dayofyr > inttemp + lmonth[i]) and (i + 1 < 12):
            inttemp = inttemp + lmonth[i]
            i = i + 1
        mon = i + 1

        day = int(dayofyr - inttemp)
        " ----------------- find hours minutes and seconds ------------"
        temp = (days - dayofyr) * 24.0
        hr = int(temp)
        temp = (temp - hr) * 60.0
        minute = int(temp)
        sec = (temp - minute) * 60.0
        return mon, day, hr, minute, sec


def ecef2enu(r_ecef, r_ref, phi_ref, lam_ref):
    """
    Convert ECEF coordinates to ENU using an ECEF reference location
    r_ref having lat = phi_ref and lon = lam_ref
    """
    # Convert lat and long angles in degress to radians
    phi_rad = phi_ref * np.pi / 180.0
    lam_rad = lam_ref * np.pi / 180.0
    # Form a 3-element column vector of the ECF (X,Y,Z) differences
    r_diff = np.array([r_ecef - r_ref]).T
    # Form the rotations transformation matrix
    A_matrix_ecef2enu = np.array([[-np.sin(lam_rad), np.cos(lam_rad), 0],
                                  [-np.sin(phi_rad) * np.cos(lam_rad), -np.sin(phi_rad) * np.sin(lam_rad),
                                   np.cos(phi_rad)],
                                  [np.cos(phi_rad) * np.cos(lam_rad), np.cos(phi_rad) * np.sin(lam_rad),
                                   np.sin(phi_rad)]])
    # Multiply the 3x3 matrix times the 3x1 column vector
    r_enu = np.dot(A_matrix_ecef2enu, r_diff)
    # Upon return flatten column vector back to a simple 1D
    # Also need to scale units of meters to what is needed
    return r_enu.flatten()


def enu2ecef(r_enu, r_ref, phi_ref, lam_ref):
    """
    Convert ENU coordinates to ECEF using an ECEF reference location
    r_ref having lat = phi_ref and lon = lam_ref
    """
    # Convert lat and long angles in degrees to radians
    phi_rad = phi_ref * np.pi / 180.0
    lam_rad = lam_ref * np.pi / 180.0
    # Form a 3-element column vector of the ENU (X,Y,Z) value
    r_enu = np.array([r_enu]).T
    # Form the rotations transformation matrix
    A_matrix_enu2ecef = np.array(
        [[-np.sin(lam_rad), -np.sin(phi_rad) * np.cos(lam_rad), np.cos(phi_rad) * np.cos(lam_rad)],
         [np.cos(lam_rad), -np.sin(phi_rad) * np.sin(lam_rad), np.cos(phi_rad) * np.sin(lam_rad)],
         [0, np.cos(phi_rad), np.sin(phi_rad)]])
    # Multiply the 3x3 matrix times the 3x1 column vector
    r_ecef = np.dot(A_matrix_enu2ecef, r_enu)
    # Add the reference to the transformation result
    r_ecef = r_ecef.flatten() + r_ref  # flatten to 1D array
    return r_ecef


def sv_user_traj_3d(gps_ds, sv_pos, user_pos, ele=10, azim=20):
    """[summary]

    Parameters:
    ----------
    GPS_ds : {[type]}
        [description]
    SV_Pos : {[type]}
        [description]
    USER_Pos : {[type]}
        [description]
    ele : {[type]}, optional
        [description] (the default is 20, which [default_description])
    azim : {[type]}, optional
        [description] (the default is 20, which [default_description])

    """

    # Mark Wickert January 2018

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    rot = 80.0 / 180 * np.pi
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b',
                    linewidth=0, alpha=0.2)
    ax.plot(sv_pos[0, 0, :], sv_pos[0, 1, :], sv_pos[0, 2, :])
    ax.plot(sv_pos[1, 0, :], sv_pos[1, 1, :], sv_pos[1, 2, :])
    ax.plot(sv_pos[2, 0, :], sv_pos[2, 1, :], sv_pos[2, 2, :])
    ax.plot(sv_pos[3, 0, :], sv_pos[3, 1, :], sv_pos[3, 2, :])
    ax.plot(user_pos[:, 0], user_pos[:, 1], user_pos[:, 2],
            'r', linewidth=3.0)
    # calculate vectors for "vertical" circle
    a = np.array([-np.sin(ele / 180 * np.pi),
                  0, np.cos(ele / 180 * np.pi)])
    b = np.array([0, 1, 0])
    b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + \
        a * np.dot(a, b) * (1 - np.cos(rot))
    ax.plot(radius * np.sin(u), radius * np.cos(u), 0, color='k',
            linestyle='dashed', alpha=0.5)
    horiz_front = np.linspace(0, np.pi, 100)
    ax.plot(radius * np.sin(horiz_front), radius * np.cos(horiz_front),
            0, color='k', alpha=0.5)
    vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    ax.plot(radius * (a[0] * np.sin(u) + b[0] * np.cos(u)),
            radius * (b[1] * np.cos(u)),
            radius * (a[2] * np.sin(u) + b[2] * np.cos(u)),
            color='k', linestyle='dashed', alpha=0.5)
    ax.plot(radius * (a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front)),
            radius * (b[1] * np.cos(vert_front)),
            radius * (a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front)),
            color='k', alpha=0.5)

    plt.legend((gps_ds.Rx_sv_list[0], gps_ds.Rx_sv_list[1],
                gps_ds.Rx_sv_list[2], gps_ds.Rx_sv_list[3], r'USER'), loc='upper right')

    ax.set_xlim3d(-2e7, 1e7)
    ax.set_ylim3d(-2e7, 1e7)
    ax.set_zlim3d(-0.75e7, 2.25e7)
    ax.set_xlabel(r'$x$ ECEF (m)')
    ax.set_ylabel(r'$y$ ECEF (m)')
    ax.set_zlabel(r'$z$ ECEF (m)')
    ax.set_title(r'SV and USER Trajectories')
    # axis('scaled')
    ax.view_init(elev=ele, azim=azim)

    print('Duration: %2.2f min' % (gps_ds.Ts * gps_ds.N_sim_steps / 60,))


def sv_user_traj_3d_interactive(gps_ds, sv_pos, user_pos, ele=10., azim=20.):
    """
    This method will provide an interactive 3d model plotted using mayavi to show all trajectories.
    :param gps_ds:
    :param sv_pos:
    :param user_pos:
    :return:
    """
    mlab.figure(1, bgcolor=(0.48, 0.48, 0.48), fgcolor=(0, 0, 0),
                size=(400, 400))
    mlab.clf()
    ##########################################################################
    # Display continents outline, using the VTK Builtin surface 'Earth'
    continents_src = BuiltinSurface(source='earth', name='Continents')
    # The on_ratio of the Earth source controls the level of detail of the
    # continents outline.
    continents_src.data_source.on_ratio = 1
    continents = mlab.pipeline.surface(continents_src, color=(0, 0, 0))

    for svn in range(0, len(sv_pos)):
        mlab.plot3d(sv_pos[svn, 0, :] / radius, sv_pos[svn, 1, :] / radius, sv_pos[svn, 2, :] / radius,
                    color=(1, 1, 0.5),
                    opacity=0.5, tube_radius=None)
        xml = len(sv_pos[svn, 0, :]) / 2
        yml = len(sv_pos[svn, 1, :]) / 2
        zml = len(sv_pos[svn, 2, :]) / 2
        xm, ym, zm = sv_pos[svn, 0, int(xml)] / radius, sv_pos[svn, 1, int(yml)] / radius, sv_pos[svn, 2, int(zml)] / radius
        label = mlab.text(xm, ym, gps_ds.Rx_sv_list[svn], z=zm, width=0.0155 * len(gps_ds.Rx_sv_list[svn]))
        label.property.shadow = True
    mlab.plot3d(user_pos[:, 0] / radius, user_pos[:, 1] / radius, user_pos[:, 2] / radius,
                color=(1, 1, 1),
                opacity=0.5, tube_radius=None)
    xml = len(user_pos[:, 0]) / 2
    yml = len(user_pos[:, 1]) / 2
    zml = len(user_pos[:, 2]) / 2
    xm, ym, zm = user_pos[int(xml), 0] / radius, user_pos[int(yml), 1] / radius, user_pos[int(zml), 2] / radius
    label = mlab.text(xm, ym, "User", z=zm, width=0.077)
    label.property.shadow = True
    ###############################################################################
    # Display a semi-transparent sphere, for the surface of the Earth

    # We use a sphere Glyph, throught the points3d mlab function, rather than
    # building the mesh ourselves, because it gives a better transparent
    # rendering.
    ocean_blue = (0.4, 0.5, 1.0)
    sphere = mlab.points3d(0, 0, 0, scale_mode='none',
                           scale_factor=2,
                           # color=(0.67, 0.77, 0.93),
                           color=ocean_blue,
                           resolution=50,
                           opacity=.85,
                           name='Earth')
    #
    # These parameters, as well as the color, where tweaked through the GUI,
    # with the record mode to produce lines of code usable in a script.
    sphere.actor.property.specular = 0.45
    sphere.actor.property.specular_power = 5
    # Backface culling is necessary for more a beautiful transparent
    # rendering.
    sphere.actor.property.backface_culling = True

    # Plot the equator and the tropiques
    theta = np.linspace(0, 2 * np.pi, 100)
    for angle in (- np.pi / 6, 0, np.pi / 6):
        x = np.cos(theta) * np.cos(angle)
        y = np.sin(theta) * np.cos(angle)
        z = np.ones_like(theta) * np.sin(angle)
        print(x)
        mlab.plot3d(x, y, z, color=(1, 1, 1),
                    opacity=0.2, tube_radius=None)

    mlab.view(azim, ele)
    mlab.show()
