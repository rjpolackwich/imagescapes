import os
import sys
import re
import random
import csv
import concurrent.futures
import collections
from collections import deque
import ujson as json
import itertools


from skyway.canvas import WNUTM5kmTiling, ProjectedUTMTiling, GeoTile, WGS84Transformer, WORLD_CRS, MAP_CRS
from skyway.utils import itrreduce, to_gjson
from skyway.query import QueryBuilder, opf, NodeWayRelationQuery
from skyway.query.nominatim import Nominatim
from skyway.query.scrape import scrape_primary_features
import tiletanic as tt

import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray as riox
import numpy as np
import shapely.ops as ops
import shapely.geometry as geom
from shapely.strtree import STRtree
import skimage.io as io

import vaex as vx

import time
from tqdm.notebook import tqdm



def iter_quadkey_paths(zone, data_path=DATA_PATH):
    for item in os.listdir(os.path.join(data_path, str(zone))):
        if qkp.match(item):
            yield qk


def setup_query():
    nwr_q = NodeWayRelationQuery()
    qb = QueryBuilder()
    qb.include_geometries()
    qb.settings.payload_format = 'json'
    qb.settings.maxsize = int(qb.settings.MAXSIZE_LIMIT / 2)
    qb.qsx.append(nwr_q)
    return qb


def setup_tiler(utm_zone):
    tiler = WNUTM5kmTiling()
    proj = ProjectedUTMTiling(zone=utm_zone, tiler=tiler)
    return proj


def fetch_osm_by_quadkey(zone, data_path):
    data_region_path = os.path.join(data_path, str(zone))
    qkdeq = collections.deque(os.listdir(data_region_path))

    node_tags = collections.defaultdict(list)
    way_tags = collections.defaultdict(list)
    rel_tags = collections.defaultdict(list)

    qb = setup_query()
    proj = setup_tiler(zone)

    pbar = tqdm(total=len(qkdeq))
    while qkdeq:
        qk = qkdeq.pop()
        tile = proj.tile_from_quadkey(qk)
        tile.toWGS84()
        west, south, east, north = tile.bounds
        qb.GlobalBoundingBox = [south, west, north, east]
        print(f"Requesting: {qk}, {tile.bounds}")
        print("\n")

        try:
            r = qb.request()
            r.raise_for_status()
            res = r.json()
        except Exception as e:
            print(f"Got exception {e}, backing off and sleeping for one minute")
            qkdeq.append(qk)
            time.sleep(60)
            continue

        nodes = [elm for elm in res['elements'] if elm['type']=='node']
        nodes_w_tags = [elm for elm in nodes if elm.get("tags") is not None]
        ways = [elm for elm in res['elements'] if elm['type']=='way']
        ways_w_tags = [elm for elm in ways if elm.get("tags") is not None]
        rels = [elm for elm in res['elements'] if elm['type']=='relation']
        rels_w_tags = [elm for elm in rels if elm.get("tags") is not None]

        n_nodes = len(nodes)
        n_nodes_tagged = len(nodes_w_tags)
        n_ways = len(ways)
        n_ways_tagged = len(ways_w_tags)
        n_rels = len(rels)
        n_rels_tagged = len(rels_w_tags)

        print(f"Total elements returned: {len(res['elements'])}")

        print(f"Num tagged/total: Nodes({n_nodes_tagged}/{n_nodes}), Ways({n_ways_tagged}/{n_ways}), Rels({n_rels_tagged}/{n_rels})")
        print(f"Total items filtered: {n_nodes_tagged + n_ways_tagged + n_rels_tagged}")

        d = {"quadkey": qk, "nodes": nodes_w_tags, "ways": ways_w_tags, "relations": rels_w_tags}
        fp = os.path.join(os.path.join(data_region_path, qk), "osm_data.json")
        with open(fp, "w") as f:
            json.dump(d, f)
        print(f"wrote OSM data to {fp}")

        for elm in nodes_w_tags:
            for key, val in elm['tags'].items():
                node_tags[key].append(val)
        for elm in ways_w_tags:
            for key, val in elm['tags'].items():
                way_tags[key].append(val)
        for elm in rels_w_tags:
            for key, val in elm['tags'].items():
                rel_tags[key].append(val)

        print("\n")
        print("\n")
        print("\n")
        pbar.update(1)
        time.sleep(5)

    return (node_tags, way_tags, rel_tags)


def build_tag_map(elm_type, zone, data_path):
    tag_map = collections.defaultdict(list)
    data_region_path = os.path.join(data_path, str(zone))
    quadkeys = [qk for qk in os.listdir(data_region_path) if "json" not in qk]

    for qk in quadkeys:
        fp = os.path.join(os.path.join(data_region_path, qk), "osm_data.json")
        with open(fp, "r") as f:
            data = json.load(f)
        elms = data[elm_type + "s"]
        for elm in elms:
            for key, val in elm['tags'].items():
                tag_map[key].append(val)
    write_path = os.path.join(data_region_path, f"{elm_type}_tags.json")
    with open(write_path, "w") as f:
        json.dump(tag_map, f)
    return tag_map


def get_children_at_zoom(parents, child_zoom):
    nodes = list()
    for parent in parents:
        if parent.zoom == child_zoom:
            return parents
        if parent.zoom > child_zoom:
            raise ValueError
        children = parent.children()
        nodes.extend(children)
    return get_children_at_zoom(nodes, child_zoom)


def get_cog_paths(zone, ard_path=DATA_PATH):
    ard_region_path = os.path.join(ard_path, str(zone))
    proj = setup_tiler(zone)
    quadkeys = [qk for qk in os.listdir(ard_region_path) if "json" not in qk]
    cog_paths = list()
    for qk in quadkeys:
        parent_tile = proj.tile_from_quadkey(qk)
        qk_ard_path = os.path.join(ard_region_path, qk)
        acq_dates = [dt for dt in os.listdir(qk_ard_path) if "json" not in dt]
        for acq_date in acq_dates:
            ard_path = os.path.join(qk_ard_path, acq_date)
            ard_items = os.listdir(ard_path)
            cogfiles = [item for item in ard_items if item[-11:] == "-visual.tif"]
            for cogfile in cogfiles:
                cog_paths.append(os.path.join(ard_path, cogfile))
    return cog_paths


def ard_image_path_from_chip(chip_fn):
    if chip_fn.endswith(".jpg"):
        chip_fn = chip_fn[:-4]
    qkhead, cat_id = chip_fn.split("_")
    qk_major = qkhead[4:16]
    cog_fn = f'''{cat_id}-visual.tif'''
    qk_cog_path = os.path.join(COG_PATH, qk_major)

    for date_dir in os.listdir(qk_cog_path):
        full_cog_path = os.path.join(os.path.join(qk_cog_path, date_dir), cog_fn)
        if os.path.exists(full_cog_path):
            return full_cog_path
    raise OSError("File not found")

