import glob
import json
import math
import os
from pprint import pprint
import sys
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib.patches import Polygon
from datetime import datetime, timedelta
import urllib.request
from matplotlib.widgets import Slider
#from mpl_toolkits.basemap import Basemap
import contextily as cx
from itertools import chain
import rasterio
from shapely.geometry import Point
import geopandas as gpd
import xarray as xr
import tarfile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from osgeo import osr, gdal
import numpy
from rasterio.plot import show as rioshow
from PIL import Image
import matplotlib.cm as cm
import numpy.ma as ma


#https://media.meteonews.net/radar/chComMET_800x618_c2/radar_20230824_1450.png
#https://media.meteonews.net/radarpre/chComMET_800x618_c2/radarpre_20230824_1635_20230825_1930.png

wn_format = np.array([
(2, str, "product"),
(2, int, "UTC day"),
(2, int, "UTC hour"),
(2, int, "UTC minute"),
(5, int, "WMO number"),
(2, int, "month"),
(2, int, "year"),
(2, str, "identifier BY"),
(10, int, "product length"), # 7 or 10?
(2, str, "identifier VS"),
(2, int, "format version"),
(2, str, "identifier SW"),
(8 + 1, str, "software version"), #1X
(2, str, "identifier PR"),
(5, str, "data precision"),
(3, str, "identifier INT"),
(4, int, "interval time"),
(2, str, "identifier GP"),
(9, str, "resolution"), #1X
(2, str, "identifier VV"),
(3 + 1, int, "prediction time"), #1X
(2, str, "identifier MF"),
(8 + 1, int, "module flags"), # 1X
(2, str, "identifier MS"),
(3, int, "text length"),
(2, str, "text")
])




def read_dwd(filename):
    #wn_format[:,0] = np.cumsum(wn_format[:,0]) # sum bytes
    sizes = np.cumsum(wn_format[:,0]) # sum bytes
    metadata = {}

    bytes_read = 0
    with open(filename, "rb") as f:
        data = b''
        while (byte := f.read(1)):
            if byte == b'\x03':
                break

            size_index = (sizes > bytes_read).nonzero()[0][0]
            current_format = wn_format[size_index]

            if sizes[size_index] - current_format[0] == bytes_read:
                data = b'' 
 
            data += byte
            bytes_read += 1

            next_format = (sizes > bytes_read).nonzero()
            if len(next_format[0]) == 0 or size_index != next_format[0][0]:
                    if current_format[1] == str:
                        metadata[current_format[2]] = data.decode("utf-8")
                    elif current_format[1] == int:
                        metadata[current_format[2]] = int(data.decode("utf-8"))
                    else:
                        metadata[current_format[2]] = data
            
        # TODO rewrite with identifier querying instead of this

        image_data = np.zeros((1200,1100))
        col = 0
        row = 0

        data = b''
        while (byte := f.read(1)):
            byte2 = f.read(1)

            if col == 1100:
                row += 1
                col = 0

            bdata = int.from_bytes(byte + byte2, byteorder='little')
            #print(bdata & 0x0FFF)
            
            
            if bdata & (1 << 13):
                # No-Data
                image_data[row, col] = 0
            else:
                image_data[row, col] = (bdata & 0x0FFF) / 4095.0
                #image_data[row, col] = ((bdata & 0x0FFF) * 0.1)/2.0 - 32.5


            col += 1

        image_data = np.flip(image_data)
        image_data = np.flip(image_data, axis = 1)
        #print(metadata)
        #print(wn_format)
        #print(np.where(np.any(wn_format[:,0] > 15)))
    return image_data



if True:
    files = glob.glob('dwd/*')
    for f in files:
        os.remove(f)

    urllib.request.urlretrieve("https://opendata.dwd.de/weather/radar/composite/wn/WN_LATEST.tar.bz2", "WN_LATEST.tar.bz2")
    tar = tarfile.open("WN_LATEST.tar.bz2", "r:bz2")  
    tar.extractall("dwd")
    tar.close()







dwd_files = sorted(os.listdir("dwd"))
image_data = read_dwd(os.path.join("dwd", dwd_files[0]))



driver = gdal.GetDriverByName( "GTiff" )
dst_ds = driver.Create( 'dst_filename.tif', 1100, 1200, 1, gdal.GDT_Float64)
dst_ds.SetGeoTransform( [ 0, 0.025, 0, 10, 0, -0.01 ] )
srs = osr.SpatialReference()
srs.ImportFromProj4('+proj=stere +lat_0=90 +lat_ts=60 +lon_0=10 +a=6378137 +b=6356752.3142451802 +no_defs +x_0=543196.83521776402 +y_0=3622588.861931001')
dst_ds.SetProjection( srs.ExportToWkt() )
raster = numpy.zeros( (1200, 1100), dtype=numpy.uint64)    
dst_ds.GetRasterBand(1).WriteArray(image_data)
# Once we're done, close properly the dataset
dst_ds = None

with rasterio.open("dst_filename.tif") as r:
     rioshow(r)


fig, ax = plt.subplots()


long = [3.551921296, 18.76728172]
lat = [45.69587048, 55.84848692]
geometry = [Point(xy) for xy in zip(long,lat)]
geo_df = gpd.GeoDataFrame(geometry = gpd.points_from_xy(long, lat))
geo_df.crs = {'init':"epsg:4326"}
geo_df.plot(ax = ax)
cx.add_basemap(ax, crs='epsg:4326', source=cx.providers.OpenStreetMap.Mapnik)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.imshow(ma.masked_array(image_data > 0, image_data), cmap=cm.jet, extent= [3.551921296, 18.76728172, 45.69587048, 55.84848692], origin='upper', interpolation='none')
fig.colorbar(im, cax=cax, orientation='vertical')




colored_image = cm.jet(image_data)
im = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
im.save("your_file.png")

#ds = gdal.Open("your_file.png")
#gt = gdal.Translate(ds, 'output.tif', outputBounds = [45.69587048, 3.551921296, 55.84848692, 18.76728172], outputSRS="EPSG:4326")


amp_0 = 0
slider_ax  = fig.add_axes([0.2, 0.025, 0.5, 0.03], facecolor="k")
slider = Slider(slider_ax, 'Time', 0.0, 24.0, valinit=amp_0, valstep=1)

image_cache = [None] * 24

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    #cur = zero + timedelta(minutes=val * 5)
    #slider.valtext.set_text(cur)
    if image_cache[int(val)] is None:
        print(int(val))
        image_data = read_dwd(os.path.join("dwd", dwd_files[int(val)]))
        image_cache[int(val)] = image_data
    else:
        image_data = image_cache[int(val)]
    im = ax.imshow(image_data, cmap='jet')
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.canvas.draw_idle()

slider.on_changed(sliders_on_changed)

plt.show()

# dd/mm/YY
now = datetime.now()
d1 = now.strftime("%Y%m%d_%H%M")
print("d1 =", d1)

#https://www.meteoschweiz.admin.ch/product/output/inca/precipitation/rate/version__20230816_1749/rate_20230816_1915.json
#https://www.meteoschweiz.admin.ch/product/output/radar/rzc/radar_rzc.20230816_1740.json


urllib.request.urlretrieve("https://www.meteoschweiz.admin.ch/product/output/versions.json", "versions.json")
version = None

with open('versions.json') as json_data:
    versions = json.load(json_data)
    version = versions["inca/precipitation/rate"]

if version is None:
    print("Could not download versions.json")
    sys.exit(0)

print(version)

backlog = now - timedelta(hours=6)
zero = datetime(year=backlog.year, month=backlog.month, day=backlog.day, hour=backlog.hour)

hours = 6 + 30

if False:
    i = 0
    for h in range(hours):
        for m in range(0, 60, 5):
            cur = zero + timedelta(hours=h, minutes=m)
            ts = cur.strftime("%Y%m%d_%H%M")
            print(cur)
            try:
                if (cur < now):
                    urllib.request.urlretrieve(f"https://www.meteoschweiz.admin.ch/product/output/radar/rzc/radar_rzc.{ts}.json", f"radar/{i}.json")
                else:
                    urllib.request.urlretrieve(f"https://www.meteoschweiz.admin.ch/product/output/inca/precipitation/rate/version__{version}/rate_{ts}.json", f"radar/{i}.json")
            except urllib.error.HTTPError:
                print("Error downloading ", ts, "past" if cur < now else "future")

            i += 1

def LV03_95toCH (e, t):
      return e - 2e6 if e >= 2e6 else e,  t - 1e6 if t >= 1e6 else t

def CHtoWGS(e, t):
      return [CHtoWGSlng(e, t), CHtoWGSlat(e, t)];

def CHtoWGSlat(e, t):
      nx, ny = LV03_95toCH(e, t)
      o = (nx - 6e5) / 1e6
      i = (ny - 2e5) / 1e6
      c = 16.9023892 + 3.238272 * i - 0.270978 * math.pow(o, 2) - 0.002528 * math.pow(i, 2) - 0.0447 * math.pow(o, 2) * i - 0.014 * math.pow(i, 3)
      c = 100 * c / 36
      return c

def CHtoWGSlng (e, t):
      nx,ny = LV03_95toCH(e, t)
      o = (nx - 6e5) / 1e6
      i = (ny - 2e5) / 1e6
      c = 2.6779094 + 4.728982 * o + 0.791484 * o * i + 0.1306 * o * math.pow(i, 2) - 0.0436 * math.pow(o, 3)
      c = 100 * c / 36
      return c

def load_json(filename):
    shapes = []
    colors = []

    with open(filename) as json_data:
        d = json.load(json_data)
        coords = d["coords"]
        x_max = coords["x_max"]
        x_min = coords["x_min"]
        y_max = coords["y_max"]
        y_min = coords["y_min"]
        x_count = coords["x_count"]
        y_count = coords["y_count"]

        for area in d["areas"]:
            color = area["color"]
            for shape in area["shapes"]:
                for subshape in shape:
                    n = subshape["i"]
                    o = subshape["j"]
                    hexcolor = "#" + str(color)
                    origin = subshape["o"]
                    direction = subshape["d"]

                    i = []

                    for c in range(len(origin)):
                        s = 0
                        a = 0
                        l = int(origin[c]) / 10.0 + 0.05
                        if n % 2 == 0:
                            s = x_min + (x_max - x_min) * (n / 2) / x_count
                            a = y_min + (y_max - y_min) * ((o - 1) / 2 + l) / y_count
                        else:
                            s = x_min + (x_max - x_min) * ((n - 1) / 2 + l) / x_count
                            a = y_min + (y_max - y_min) * (o / 2) / y_count

                        u, d = CHtoWGS(1e3 * s, 1e3 * a)
                        i.append([u, d])
                        if (2 * c < len(direction)):
                            n += ord(direction[2 * c]) - 77
                            o += ord(direction[2 * c + 1]) - 77
                        else:
                            break

                    shapes.append(i)
                    colors.append(hexcolor)

    return shapes, colors
       
fig,ax = plt.subplots()

long = [4.0, 13.0]
lat = [45.0, 49.0]
geometry = [Point(xy) for xy in zip(long,lat)]
geo_df = gpd.GeoDataFrame(geometry = gpd.points_from_xy(long, lat))
geo_df.crs = {'init':"epsg:4326"}

geo_df.plot(ax = ax)
cx.add_basemap(ax, crs='epsg:4326', source=cx.providers.OpenStreetMap.Mapnik)


shapes, colors = load_json('meteoswiss.json')
for i in range(len(shapes)):
    vertices = shapes[i]
    y = np.array(vertices)
    p = Polygon(y, facecolor = colors[i])
    ax.add_patch(p)

amp_0 = 12 * 6

ax.set_xlim([4, 13])
ax.set_ylim([45, 49])

amp_slider_ax  = fig.add_axes([0.2, 0.025, 0.5, 0.03], facecolor="k")
amp_slider = Slider(amp_slider_ax, 'Time', 0.0, hours * 12, valinit=amp_0, valstep=1)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    ax.patches = []
    cur = zero + timedelta(minutes=val * 5)
    amp_slider.valtext.set_text(cur)
    shapes, colors = load_json(f"radar/{int(amp_slider.val)}.json")
    for i in range(len(shapes)):
        vertices = shapes[i]
        y = np.array(vertices)
        p = Polygon(y, facecolor = colors[i])
        ax.add_patch(p)
    fig.canvas.draw_idle()

amp_slider.on_changed(sliders_on_changed)

plt.show()

flashes = []
with open('meteoswiss_lightning.json') as json_data:
    d = json.load(json_data)
    for cluster in d:
        #print(cluster)
        for item in d[cluster]:
            flashes.append([float(item[0]), float(item[1])])
            #print(item)

flashes = np.array(flashes, dtype=object)

plt.scatter(flashes[:,0], flashes[:,1], s=0.1)
plt.show()