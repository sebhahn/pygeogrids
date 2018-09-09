# Copyright (c) 2018, TU Wien, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of TU Wien, Department of Geodesy and Geoinformation
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL TU WIEN, DEPARTMENT OF GEODESY AND
# GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import unittest
import numpy as np

from pygeogrids.geodetic_datum import GeodeticDatum
from pygeogrids.nearest_neighbor import findGeoNN


class BasicGrid(object):

    def __init__(self, lon, lat, loc=None, geodatum='WGS84',
                 setup_kdTree=True):

        self.lon = np.asanyarray(lon)
        self.lat = np.asanyarray(lat)
        self.parent = None

        if self.lat.shape != self.lon.shape:
            raise ValueError("lat and lon have not equal shapes")

        if loc is None:
            self.loc = np.arange(self.lon.size, dtype=np.int32)
        else:
            self.loc = np.asanyarray(loc)

        self.iloc = np.zeros(self.loc.max() + 1, dtype=np.int32)
        self.iloc[self.loc] = np.arange(self.loc.size)

        self.geodatum = GeodeticDatum(geodatum)
        self.kdTree = None

        if setup_kdTree:
            self._setup_kdtree()

    def _setup_kdtree(self):
        """
        Setup kdTree for searching nearest neighbors.
        """
        if self.kdTree is None:
            self.kdTree = findGeoNN(self.lon, self.lat, self.geodatum)
            self.kdTree._build_kdtree()

    def __str__(self):
        """
        Print grid information.
        """
        info = 'Grid information\n'
        info += 'loc.shape: {}\n'.format(self.loc.shape)
        info += 'lon.shape: {}\n'.format(self.lon.shape)
        info += 'lat.shape: {}'.format(self.lat.shape)

        return info

    def __getitem__(self, iloc):
        """
        Get item.
        """
        return BasicSubGrid(self, self.loc[iloc])

    def __iter__(self):
        self._idx_loc = 0
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        """
        Yields all grid points in location id order.

        Returns
        -------
        loc : int32
            Location id.
        lon : float
            Longitude coordinate.
        lat : float
            Latitude coordinate.
        """
        if self._idx_loc < self.loc.size:
            loc = self.loc[self._idx_loc]
            lon = self.lon[self.iloc[loc]]
            lat = self.lat[self.iloc[loc]]
            self._idx_loc += 1
            return loc, lon, lat
        else:
            raise StopIteration

    def grid_points(self, bbox=None):
        """
        Return grid points.

        Returns
        -------
        loc : numpy.ndarray
            Location ids.
        lon : numpy.ndarray
            Longitude coordinates.
        lat : numpy.ndarray
            Latitude coordinates.
        """
        return self.loc, self.lon, self.lat

    def loc2lonlat(self, loc):
        """
        Longitude and latitude for given location id.

        Parameters
        ----------
        loc : int32 or iterable
            Location id.

        Returns
        -------
        lon : float
            Longitude coordinate.
        lat : float
            Latitude coordinate.
        """
        return self.lon[self.iloc[loc]], self.lat[self.iloc[loc]]

    def subgrid_from_loc(self, loc):
        pass

    def subgrid_from_bbox(self, bbox):
        pass

    def subgrid_from_shp(self, shp):
        pass


class BasicSubGrid(BasicGrid):

    def __init__(self, parent, loc):
        self.parent = parent
        self.child_loc = loc

    @property
    def loc(self):
        return self.child_loc

    @property
    def iloc(self):
        return self.parent.iloc

    @property
    def lon(self):
        return self.parent.lon[self.iloc[self.loc]]

    @property
    def lat(self):
        return self.parent.lat[self.iloc[self.loc]]

    @property
    def geodatum(self):
        return self.parent.geodatum


class Basic2dGrid(object):

    def __init__(self, x, y, loc=None, geodatum='WGS84'):

        self.x = np.asanyarray(x)
        self.y = np.asanyarray(y)
        self.geodatum = GeodeticDatum(geodatum)
        self.parent = None

    def __str__(self):
        """
        Print grid information.
        """
        info = 'Grid information\n'
        info += 'x.shape: {}\n'.format(self.x.shape)
        info += 'y.shape: {}'.format(self.y.shape)

        return info

    def __getitem__(self, rows, cols):
        """
        Get item.
        """
        pass

    def loc2rowcol(self, loc):
        pass

    def loc2xy(self, loc):
        pass


# class Test_BasicGrid(unittest.TestCase):

#     def setUp(self):
#         """
#         Setup dummy grid.
#         """
#         lon = np.arange(-180, 180, 30)
#         lat = np.arange(-90, 90, 30)
#         self.lon, self.lat = np.meshgrid(lon, lat)
#         self.loc = np.arange(0, self.lon.size) * 10

#     def test_grid_creation(self):
#         """
#         Test grid creation and subgrids.
#         """
#         grid = BasicGrid(self.lon.flatten(), self.lat.flatten(), self.loc)
#         subgrid = grid[np.arange(10)]
#         print(subgrid)
#         subsubgrid = subgrid[np.arange(5)]
#         print(subsubgrid)

#         for loc, lon, lat in subsubgrid:
#             print(loc, lon, lat)
#             lo, la = subsubgrid.loc2lonlat(loc)
#             print(lon == lo, lat == la)


class Test_Basic2dGrid(unittest.TestCase):

    def setUp(self):
        """
        Setup dummy grid.
        """
        self.x = np.arange(-180, 180, 30)
        self.y = np.arange(-90, 90, 30)

    def test_grid_creation(self):
        """
        Test 2d grid creation and subgrids.
        """
        grid = Basic2dGrid(self.x, self.y)
        print(grid)


if __name__ == '__main__':
    unittest.main()
