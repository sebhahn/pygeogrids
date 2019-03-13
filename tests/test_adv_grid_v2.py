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
                 setup_kdTree=True, parent=None):

        self.parent = parent

        if self.parent is None:
            self._lon = np.asanyarray(lon)
            self._lat = np.asanyarray(lat)

            if self.lat.shape != self.lon.shape:
                raise ValueError("lat and lon have not equal shapes")

            if loc is None:
                self.loc = np.arange(self.lon.size, dtype=np.int32)
                self.iloc = self.loc
            else:
                self.loc = np.asanyarray(loc)

                # use a dict instead?
                # create look-up table
                self.iloc = np.zeros(self.loc.max() + 1, dtype=np.int32)
                self.iloc[self.loc] = np.arange(self.loc.size)
        else:
            # Child grid is about to be initialized
            self._lon = None
            self._lat = None
            self._iloc = None
            self.loc = loc

        self._geodatum = GeodeticDatum(geodatum)
        self.kdTree = None
        self._iter_loc = 0

        if setup_kdTree:
            self._setup_kdtree()

    @property
    def lon(self):
        if self.parent is None:
            return self._lon
        else:
            return self.parent.lon[self.iloc[self.loc]]

    @property
    def lat(self):
        if self.parent is None:
            return self._lat
        else:
            return self.parent.lat[self.iloc[self.loc]]

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, value):
        self._loc = value

    @property
    def iloc(self):
        if self.parent is None:
            return self._iloc
        else:
            return self.parent.iloc

    @iloc.setter
    def iloc(self, value):
        self._iloc = value

    @property
    def geodatum(self):
        if self.parent is None:
            return self._geodatum
        else:
            return self.parent.geodatum

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
        info = 'Grid information: '
        info += 'loc.shape: {}, '.format(self.loc.shape)
        info += 'lon.shape: {}, '.format(self.lon.shape)
        info += 'lat.shape: {}.'.format(self.lat.shape)

        return info

    @classmethod
    def _create_subgrid(cls, self_obj, loc):
        """
        Create subgrid just by storing the location id.
        All other attributes (lon, lat, etc.) are accessed via
        the parent grid.

        Parameters
        ----------
        self_obj : BasicGrid
            Current (i.e. future parent) grid object.
        loc : numpy.ndarray
            List of location ids for subgrid.

        Returns
        -------
        grid : BasicGrid
            A 'sub' BasicGrid object with access to parent grid object.
        """
        return cls(None, None, loc, parent=self_obj)

    def __eq__(self, other):
        """
        Compare lon, lat and loc.

        Returns
        -------
        result : boolean
            Returns True if grids are equal.
        """
        if self.geodatum.name == other.geodatum.name:
            geosame = True
        else:
            geosame = False

        # return np.all([lonsame, latsame, gpisame, subsetsame, shapesame,
        #                geosame])

    def __getitem__(self, iloc):
        """
        Get item.
        """
        return BasicGrid._create_subgrid(self, self.loc[iloc])

    def __iter__(self):
        self._iter_loc = 0
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
        if self._iter_loc < self.loc.size:
            loc = self.loc[self._iter_loc]
            lon = self.lon[self.iloc[loc]]
            lat = self.lat[self.iloc[loc]]
            self._iter_loc += 1
            return loc, lon, lat
        else:
            raise StopIteration

    def loc2lonlat(self, loc):
        """
        Longitude and latitude for given location id(s).

        Parameters
        ----------
        loc : int or numpy.ndarray
            Location id.

        Returns
        -------
        lon : float or numpy.ndarray
            Longitude coordinate.
        lat : float or numpy.ndarray
            Latitude coordinate.
        """
        return self.lon[self.iloc[loc]], self.lat[self.iloc[loc]]

    def nn_loc(self, lon, lat, max_dist=np.Inf):
        """
        Finds nearest location id, builds kdTree if it does not yet exist.

        Parameters
        ----------
        lon : float or numpy.ndarray
            Longitude of point.
        lat : float or numpy.ndarray
            Latitude of point.

        Returns
        -------
        loc : int
            Grid point index.
        distance : float
            Distance of found location to given lon, lat.
            At the moment not on a great circle but in spherical
            cartesian coordinates.
        """
        self._setup_kdtree()

        dist, ind = self.kdTree.find_nearest_index(lon, lat,
                                                   max_dist=max_dist)

        return self.loc[ind], self.lon[ind], self.lat[ind], dist

    def subgrid(self, loc=None, bbox=None):
        """
        Select a subgrid based on location ids, geographical bounding box or
        shape file.

        Parameters
        ----------
        loc : numpy.ndarray, optional
            List of location ids for subgrid.
        bbox : tuple, optional

        Returns
        -------
        grid : BasicGrid
            A 'sub' BasicGrid object with access to parent grid object.
        """
        if bbox is not None:
            llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = bbox
            index = np.where((self.lat <= urcrnrlat) &
                             (self.lat >= llcrnrlat) &
                             (self.lon <= urcrnrlon) &
                             (self.lon >= llcrnrlon))
            loc = index

        return self[loc]

    def to_cell_grid(self, cellsize=5.0, cellsize_lat=None, cellsize_lon=None):
        """
        Create a CellGrid from BasicGrid.

        Parameters
        ----------
        cellsize : float, optional
            Cell size in degrees
        cellsize_lon : float, optional
            Cell size in degrees on the longitude axis
        cellsize_lat : float, optional
            Cell size in degrees on the latitude axis

        Returns
        -------
        cell_grid : CellGrid object
            Cell grid object.
        """
        pass


class CellGrid(BasicGrid):

    def __init__(self, lons, lats, cells, locs=None, geodatum='WGS84',
                 setup_kdTree=False, **kwargs):

        super(CellGrid, self).__init__(lons, lats, locs, geodatum=geodatum,
                                       setup_kdTree=setup_kdTree, **kwargs)

        self._cells = np.asanyarray(cells)

    def __eq__(self, other):
        """
        Compare if grids are equal.

        Returns
        -------
        result : boolean
            Returns true if equal.
        """
        basicsame = super(CellGrid, self).__eq__(other)
        cellsame = True

        return np.all([basicsame, cellsame])

    def loc2cell(locs):
        cells = locs
        return cells

    def subgrid(locs=None, bbox=None, cells=None):
        pass


class Test_BasicGrid(unittest.TestCase):

    def setUp(self):
        """
        Setup dummy grid.
        """
        lon = np.arange(-180, 180, 30)
        lat = np.arange(-90, 90, 30)
        self.lon, self.lat = np.meshgrid(lon, lat)
        self.loc = np.arange(0, self.lon.size) * 10

    def test_grid_creation(self):
        """
        Test grid creation and subgrids.
        """
        grid = BasicGrid(self.lon.flatten(), self.lat.flatten(), self.loc)
        subgrid = grid[np.arange(10)]
        print(subgrid)
        subsubgrid = subgrid[np.arange(5)]
        print(subsubgrid)

        for loc, lon, lat in subgrid:
            print(loc, lon, lat)
            lo, la = subgrid.loc2lonlat(loc)
            print(lon == lo, lat == la)

        for loc, lon, lat in subsubgrid:
            print(loc, lon, lat)
            lo, la = subsubgrid.loc2lonlat(loc)
            print(lon == lo, lat == la)


if __name__ == '__main__':
    # unittest.main()

    lon = np.arange(-40, 44, 4)
    lat = np.arange(-40, 44, 4)
    lon, lat = np.meshgrid(lon, lat)
    loc = np.arange(0, lon.size) * 10

    grid = BasicGrid(lon.flatten(), lat.flatten(), loc)
    subgrid = grid[np.arange(10)]
    # print(subgrid.loc, subgrid.lon, subgrid.lat)
    subsubgrid = subgrid[np.arange(5)]
    print(subsubgrid.loc, subsubgrid.lon, subsubgrid.lat)

    subsubgrid2 = subgrid.subgrid(np.arange(5))
    print(subsubgrid2.loc, subsubgrid2.lon, subsubgrid2.lat)

    # print(subgrid.nn_loc(0, -39))
    # print(subsubgrid.nn_loc(0, -39))

    # print(subsubgrid, subsubgrid.lon)
    # print(subsubgrid.parent == subgrid)
    # print(subsubgrid.parent.parent == grid)

    # think about storing location properties, datastructure dict?
    # change overall data structure holding lon/lat/loc
