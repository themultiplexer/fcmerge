import json
import math
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
from shapely.geometry import Point
import geopandas as gpd
import xarray as xr


wn_format = [
("A2", "product"),
("I2", "UTC day"),
("I2", "UTC hour"),
("I2", "UTC minute"),
("I5", "WMO number"),
("I2", "month"),
("I2", "year"),
("A2", "identifier BY"),
("I7", "product length"),
("A2", "identifier VS"),
("I2", "format version"),
("A2", "identifier SW"),
("A8", "software version"),
("A2", "identifier PR"),
("A4", "data precision"),
("A3", "identifier INT"),
("I4", "interval time"),
("A2", "identifier GP"),
("A9", "resolution"),
("A2", "identifier VV"),
("I3", "prediction time"),
("A2", "identifier MF"),
("I8", "module flags"),
("A2", "identifier MS"),
("I3", "text length"),
#("m * A1", "text length")
]

bytes_read = 0

with open("WN2308170035_000", "rb") as f:
    while (byte := f.read(1)):
        if byte == b'\x03':
            break
        print(byte)
        bytes_read += 1

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

if version == None:
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

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


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
                    #plt.plot(subshape["i"], subshape["j"])
                    n = subshape["i"]
                    o = subshape["j"]
                    hexcolor = "#" + str(color)
                    #print(subshape["i"], subshape["j"], "#" + str(color))
                    os = subshape["o"]
                    ds = subshape["d"]

                    #print(subshape["d"], subshape["o"])

                    i = []

                    for c in range(len(os)):
                        s = 0
                        a = 0
                        l = int(os[c]) / 10.0 + 0.05
                        if n % 2 == 0:
                            s = x_min + (x_max - x_min) * (n / 2) / x_count
                            a = y_min + (y_max - y_min) * ((o - 1) / 2 + l) / y_count
                        else:
                            s = x_min + (x_max - x_min) * ((n - 1) / 2 + l) / x_count
                            a = y_min + (y_max - y_min) * (o / 2) / y_count

                        u, d = CHtoWGS(1e3 * s, 1e3 * a)
                        i.append([u, d])
                        if (2 * c < len(ds)):
                            n += ord(ds[2 * c]) - 77
                            o += ord(ds[2 * c + 1]) - 77
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
amp_slider = Slider(amp_slider_ax, 'Amp', 0.0, hours * 12, valinit=amp_0, valstep=1)


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