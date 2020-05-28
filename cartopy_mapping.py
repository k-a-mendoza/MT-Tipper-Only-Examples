
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy
import matplotlib.pyplot as plt
import pyproj
import os, glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import patheffects
import matplotlib.ticker as mticker

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np


from owslib.wmts import TileMatrixSetLink, TileMatrixLimits, _TILE_MATRIX_SET_TAG, _TILE_MATRIX_SET_LIMITS_TAG, _TILE_MATRIX_LIMITS_TAG

def custom_from_elements(link_elements):
    links = []
    for link_element in link_elements:
        matrix_set_elements = link_element.findall(_TILE_MATRIX_SET_TAG)
        if len(matrix_set_elements) == 0:
            raise ValueError('Missing TileMatrixSet in %s' % link_element)
        elif len(matrix_set_elements) > 1:
            set_limits_elements = link_element.findall(
                _TILE_MATRIX_SET_LIMITS_TAG)
            if set_limits_elements:
                raise ValueError('Multiple instances of TileMatrixSet'
                                  ' plus TileMatrixSetLimits in %s' %
                                  link_element)
            for matrix_set_element in matrix_set_elements:
                uri = matrix_set_element.text.strip()
                links.append(TileMatrixSetLink(uri))
        else:
            uri = matrix_set_elements[0].text.strip()

            tilematrixlimits = {}
            path = '%s/%s' % (_TILE_MATRIX_SET_LIMITS_TAG,
                              _TILE_MATRIX_LIMITS_TAG)
            for limits_element in link_element.findall(path):
                tml = TileMatrixLimits(limits_element)
                if tml.tilematrix:
                    tilematrixlimits[tml.tilematrix] = tml

            links.append(TileMatrixSetLink(uri, tilematrixlimits))
    return links

TileMatrixSetLink.from_elements = custom_from_elements

def plot_map(df,show=True):
    # make figure look nice
    extent = (-139.2, -139.1, 61.325, 61.385)
    fig    = plt.figure(figsize=(15,15))
    ax     = fig.add_axes([0, 0, 1, 1],projection=ccrs.PlateCarree())
    ax.set_extent(extent, ccrs.Geodetic())
    # add color shaded relief
    url    = 'https://map1.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
    layer  = 'ASTER_GDEM_Color_Shaded_Relief'
    ax.add_wmts(url, layer,alpha=0.8)
    # plot data
    plt.scatter(df['Long'],df['Lat'],marker='o',
                s=300,color='blue',edgecolor='black',label='Station',zorder=4)
    
    # plot station names
    x_label_offset=0.00125
    y_label_offset=-0.0025
    for x, y, label in zip(df['Long'],df['Lat'], df['Station']):
        ax.text(x+x_label_offset, y+y_label_offset, label,
                ha='left',va='bottom',color='white', size=15)
    # add lat-lon lines.
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.8, color='black')
    gl.xlocator = mticker.FixedLocator(np.linspace(extent[0],extent[1],num=5))
    gl.ylocator = mticker.FixedLocator(np.linspace(extent[2],extent[3],num=5))
    gl.xlabels_bottom=True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabels_left=True
    gl.xlabel_style = {'color': 'black','size': 15}
    gl.ylabel_style = {'color': 'black','size': 15}
    if show:
        plt.show()
    return ax


def plot_map_mt_objs(mt_objs,show=True):
    # parse data objects
    coords = [(mt_obj.lat, mt_obj.lon ,mt_obj.station) for mt_obj in mt_objs]
    coords = np.asarray(coords,dtype=np.float64).T
    extent = (-139.2, -139.1, 61.325, 61.385)
    
    fig    = plt.figure(figsize=(15,15))
    ax     = fig.add_axes([0, 0, 1, 1],projection=ccrs.PlateCarree())
    ax.set_extent(extent, ccrs.Geodetic())
    # add color shaded relief
    url    = 'https://map1.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
    layer  = 'ASTER_GDEM_Color_Shaded_Relief'
    ax.add_wmts(url, layer,alpha=0.8)
    # plot data
    plt.scatter(coords[1,:],coords[0,:],marker='o',
                s=300,color='blue',edgecolor='black',label='Station',zorder=4)
    
    # plot station names
    x_label_offset=0.00125
    y_label_offset=-0.0025
    for x, y, label in zip(coords[1,:],coords[0,:],coords[2,:]):
        str_label = str(int(float(label)))
        x_coord   = float(x) + x_label_offset
        y_coord   = float(y) + y_label_offset
        ax.text(x_coord, y_coord, str_label,ha='left',va='bottom',color='white', size=15)
    # add lat-lon lines.
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.8, color='black')
    gl.xlocator = mticker.FixedLocator(np.linspace(extent[0],extent[1],num=5))
    gl.ylocator = mticker.FixedLocator(np.linspace(extent[2],extent[3],num=5))
    gl.xlabels_bottom=True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabels_left=True
    gl.xlabel_style = {'color': 'black','size': 15}
    gl.ylabel_style = {'color': 'black','size': 15}
    if show:
        plt.show()
    return ax

    
def convert_wgs_to_utm(origin):
    # from https://stackoverflow.com/questions/40132542/get-a-cartesian-projection-accurate-around-a-lat-lng-pair
    utm_band = str(int((np.floor((origin[0] + 180) / 6 ) % 60) + 1))
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if origin[1] >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code

    
def plot_induction_arrows(ax,df,color='white',scale=1000,show=True,zorder=4):
    this_df           = df.copy()
    this_df           = this_df.dropna(axis=0).reset_index()
    SCALE             = scale
    origin            = (this_df['Long'].values[0], this_df['Lat'].values[0])
    
    utm_code = convert_wgs_to_utm(origin)
    crs_wgs = pyproj.Proj(init='epsg:4326',) # assuming you're using WGS84 geographic
    crs_utm = pyproj.Proj(init='epsg:{0}'.format(utm_code))
    
    east_0, north_0 = pyproj.transform(crs_wgs, crs_utm, 
                                       this_df['Long'].values, this_df['Lat'].values,
                                       always_xy='True')
    east, north = pyproj.transform(crs_utm, crs_wgs,
                                   this_df['East'].values*SCALE +east_0, this_df['North'].values*SCALE +north_0,
                                   always_xy='True')
    
    east = np.asarray(east)  - this_df['Long']
    north= np.asarray(north) - this_df['Lat']
    
    for index, row in this_df.iterrows():
        ax.arrow(row['Long'],row['Lat'],east[index],north[index],zorder=zorder,
                 facecolor=color,head_width=0.001,head_length=0.002,width=0.0006)
    if show:
        plt.show()
