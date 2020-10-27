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



def early_osm_inference():
    with open("/home/ubuntu/data/ard/33/node_tags.json") as f:
        node_tags = json.load(f)

    with open("/home/ubuntu/data/ard/33/way_tags.json") as f:
        way_tags = json.load(f)

    with open("/home/ubuntu/data/ard/33/rel_tags.json") as f:
        rel_tags = json.load(f)

    ntag_counts = {key: collections.Counter(val) for key, val in node_tags.items()}
    wtag_counts = {key: collections.Counter(val) for key, val in way_tags.items()}
    rtag_counts = {key: collections.Counter(val) for key, val in rel_tags.items()}

    node_keycounts = sorted([(key, len(val)) for key, val in node_tags.items()], key=lambda t: t[-1], reverse=True)
    way_keycounts = sorted([(key, len(val)) for key, val in way_tags.items()], key=lambda t: t[-1], reverse=True)
    rel_keycounts = sorted([(key, len(val)) for key, val in rel_tags.items()], key=lambda t: t[-1], reverse=True)

    #wtags_ordered = sorted([(key, wtag_counts[key].most_common(1)) for key, val in wtag_counts.items()], key=lambda t: t[-1][-1][-1], reverse=True)
    #wtags_ordered[:50]


## Load nodata scores map
def load_nodata_scores():
    with open("/home/ubuntu/data/chips/33/nodata_index.json", "r") as f:
        ndix = json.load(f)
    return ndix



def osm_tables_to_kv(tables):
    legal = collections.defaultdict(lambda: collections.defaultdict(list))
    restricted = collections.defaultdict(lambda: collections.defaultdict(list))
    for topic, kvd in tables.items():
        for key, val_list in kvd.items():
            for val in val_list:
                try:
                    tag_val, desc = val
                    if " " in tag_val and tag_val.lower() != "user defined":
                        restricted[topic][key].append(tag_val)
                    else:
                        legal[topic][key].append(tag_val)
                except ValueError:
                    restricted[topic][key].append(val)
    return legal, restricted


def discriminate_tagschemes():
    kvd_legal, kvd_restricted = osm_tables_to_kv(tables)
    kvd_legal.update(tags)


def get_all_tagkeys(keytag_dict):
    tagkeys = list()
    for key, val_list in keytag_dict.items():
        for val in val_list:
            tagkeys.append(val)
    return tagkeys


#tagkeys_legal = get_all_tagkeys(kvd_legal)
def compile_all_tagschema():
    all_tagkeys = [t for t in tagkeys_legal]
    for topic, keydict in kvd_restricted.items():
        all_tagkeys.extend(list(keydict.keys()))

    kvd_all = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in [kvd_legal, kvd_restricted]:
        for topic, keydict in d.items():
            for tagkey, tagvals in keydict.items():
                for tagval in tagvals:
                    kvd_all[topic][tagkey].append(tagval)
    return kvd_all

def comile_all_primary_keys(kvd_all):
    all_primary_keys = list()
    for topic, keydict in kvd_all.items():
        for key in keydict:
            all_primary_keys.append(key)
    return all_primary_keys


def primary_key_element_stats(all_primary_keys):
    ntag_counts_f = {k: v for k, v in ntag_counts.items() if k in all_primary_keys}
    wtag_counts_f = {k: v for k, v in wtag_counts.items() if k in all_primary_keys}
    rtag_counts_f = {k: v for k, v in rtag_counts.items() if k in all_primary_keys}
    return (ntag_counts_f, wtag_counts_f, rtag_counts_f)


def tabulate_element_primary_stats(outfile="/home/ubuntu/data/osm/primary_keytags.csv", overwrite=False):
    if os.path.exists(outfile) and not overwrite:
        return False
    with open(outfile, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Topic", "TagKey", "NodeCount", "WayCount", "RelCount"])
        for topic, keydict in kvd_all.items():
            for key in keydict.keys():
                nc, wc, rc = (0, 0, 0)
                if ntag_counts_f.get(key):
                    nc = sum([t[-1] for t in ntag_counts_f[key].items()])
                if wtag_counts_f.get(key):
                    wc = sum([t[-1] for t in wtag_counts_f[key].items()])
                if rtag_counts_f.get(key):
                    rc = sum([t[-1] for t in rtag_counts_f[key].items()])
                writer.writerow([topic, key, nc, wc, rc])


def load_dataset_with_filtercol(infile="/home/ubuntu/data/osm/primary_keytags.csv"):
    key_dist = pd.read_csv(infile)
    default_selections = [True]*len(key_dist)
    key_dist['Selected'] = default_selections
    return key_dist


def filter_tagkeys_from_selection(key_dist_f):
    key_dist_f[key_dist_f['Selected'] == True]
    all_filtered_keys = list(key_dist_filtered.TagKey.values)
    return all_filtered_keys


def load_complete_primary_features_description(infile="/home/ubuntu/data/osm/primary_features.json"):
    with open(infile) as f:
        return json.load(f)


def filter_osm_by_region(osm_keys, out_file="primary_osm_data_filtered.json", in_file="osm_data.json", data_path=COG_PATH):
    quadkeys = [item for item in os.listdir(data_path) if qkp.match(item)]
    for qk in tqdm(quadkeys):
        print(qk)
        infile = os.path.join(os.path.join(data_path, qk), in_file)
        outfile = os.path.join(os.path.join(data_path, qk), out_file)
        filter_osm_data(osm_keys, qk, infile, outfile)
        print("\n")


def filter_osm_data(osm_keys, quadkey, infile, outfile):
    d = dict()
    d['quadkey'] = quadkey
    d['nodes'] = list()
    d['ways'] = list()
    d['relations'] = list()

    all_keys_set = set(osm_keys)

    with open(infile) as f:
        raw = json.load(f)
        for elm_type in ['nodes', "ways", "relations"]:
            raw_elms = raw[elm_type]
            print(f'''raw {elm_type} length: {len(raw_elms)}''')
            for elm in raw_elms:
                tags = elm.get('tags')
                elm_keys_set = set(list(tags.keys()))
                primary_intr = all_keys_set.intersection(elm_keys_set)
                if len(primary_intr) > 0:
                    new_elm = dict()
                    new_elm['type'] = elm['type']
                    new_elm['id'] = elm['id']
                    new_elm['tags'] = [{tk: elm['tags'][tk] for tk in primary_intr}]
                    if elm['type'] == "node":
                        new_elm['lat'] = elm['lat']
                        new_elm['lon'] = elm['lon']
                    if elm['type'] == 'way':
                        new_elm['geometry'] = elm['geometry']
                    if elm['type'] == 'relation':
                        new_elm['members'] = elm['members']
                    d[elm_type].append(elm)


    with open(outfile, "w") as ff:
        json.dump(d, ff)

    print(f'''filtered nodes length: {len(d['nodes'])}''')
    print(f'''filtered ways length: {len(d['ways'])}''')
    print(f'''filtered relations length: {len(d['relations'])}''')
