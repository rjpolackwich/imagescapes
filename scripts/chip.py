import os
import ujson as json
from multiprocessing import Process, Pool, Queue, current_process
import concurrent.futures

import click
from tqdm import tqdm

import numpy as np
import csv
import rasterio
import skimage.io as io
from skyway.canvas import WNUTM5kmTiling, ProjectedUTMTiling

ARD_BASE_PATH = "/home/ubuntu/data/ard"
DFT_ZONE = 33
ARD_REGION_PATH = os.path.join(ARD_BASE_PATH, str(DFT_ZONE))

def setup_tiler(zone):
    tiler = WNUTM5kmTiling()
    proj = ProjectedUTMTiling(zone=zone, tiler=tiler)
    return proj

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


def get_cog_paths(zone=33, ard_path="/home/ubuntu/data/ard"):
    ard_region_path = os.path.join(ard_path, str(zone))
    quadkeys = [qk for qk in os.listdir(ard_region_path) if "json" not in qk]
    cog_paths = list()
    for qk in quadkeys:
        qk_ard_path = os.path.join(ard_region_path, qk)
        acq_dates = [dt for dt in os.listdir(qk_ard_path) if "json" not in dt]
        for acq_date in acq_dates:
            ard_path = os.path.join(qk_ard_path, acq_date)
            ard_items = os.listdir(ard_path)
            cogfiles = [item for item in ard_items if item[-11:] == "-visual.tif"]
            for cogfile in cogfiles:
                cog_paths.append(os.path.join(ard_path, cogfile))
    return cog_paths


def get_cog_path_from_chip(chip_fn, zone=33):
    ard_region_path = os.path.join(ARD_BASE_PATH, str(zone))
    if chip_fn.endswith(".jpg"):
        chip_fn = chip_fn[:-4]
    qkhead, cat_id = chip_fn.split("_")
    qk_major = qkhead[4:16]
    cog_fn = f'''{cat_id}-visual.tif'''
    qk_cog_path = os.path.join(ard_region_path, qk_major)

    for date_dir in os.listdir(qk_cog_path):
        full_cog_path = os.path.join(os.path.join(qk_cog_path, date_dir), cog_fn)
        if os.path.exists(full_cog_path):
            return full_cog_path
    raise OSError(f"ARD parent file for: {chip_fn} not found")


def write_chips(ard_path, dst_zoom=17, base_write_dir="/home/ubuntu/data/chips"):
    # get cat_id and qk from input cog filepath
    head, cogfilename = os.path.split(ard_path)
    head, acq_date = os.path.split(head)
    head, quadkey_major = os.path.split(head)
    head, zone = os.path.split(head)

    proj = setup_tiler(int(zone))
    zone_write_dir = os.path.join(base_write_dir, zone)

    cat_id = cogfilename[:-11]

    # get qk children
    parent_tile = proj.tile_from_quadkey(quadkey_major)
    child_tiles = get_children_at_zoom([parent_tile], dst_zoom)

    with rasterio.open(ard_path) as src:
        for child in child_tiles:
            chip_write_path = os.path.join(zone_write_dir, f"Z{zone}-{child.quadkey}_{cat_id}.jpg")
            with rasterio.open(ard_path) as src:
                window = rasterio.windows.from_bounds(*child.bounds, transform=src.transform)
                with rasterio.open(chip_write_path, "w",
                        driver="jpeg",
                        width=window.width,
                        height=window.height,
                        count=src.count,
                        dtype=src.profile['dtype']) as dst:
                    dst.write(src.read(window=window))
    return True


def nodata_test(arr):
    upper_left = np.sum(arr[0, 0, ...])
    upper_right = np.sum(arr[0, -1, ...])
    lower_left = np.sum(arr[-1, 0, ...])
    lower_right = np.sum(arr[-1, -1, ...])

    corner_sums = [upper_left, upper_right, lower_left, lower_right]
    if all(corner_sums): return "all_data"
    if not any(corner_sums): return "no_data"
    return "partial_data"


def nodata_batch(filepaths, pathdir):
    return [nodata_test(io.imread(os.path.join(pathdir, fp))) for fp in filepaths]


def paths_in_chunks(pathdir, chunksize=20):
    filenames = os.listdir(pathdir)
    while filenames:
        chunk = []
        while len(chunk) < chunksize:
            try:
                fn = filenames.pop()
                if not fn.endswith(".jpg"):
                    click.echo(f'''Non-image file: {fn}''')
                    continue
            except IndexError:
                break
            chunk.append(fn)
        if chunk:
            yield chunk


@click.command()
@click.option("--filename", default="nodata_index.json", help="output nodata filename")
@click.option("--chip_dir", default="/home/ubuntu/data/chips/33", help="directory with chips to test")
@click.option("--overwrite", default=True, help="write over existing file")
def index_nodata(filename, chip_dir, overwrite):
#    click.echo(chip_dir)
#    return
    nodata_scorebook = dict()
    nodata_scorebook["all_data"] = list()
    nodata_scorebook["partial_data"] = list()
    nodata_scorebook["no_data"] = list()

    chip_qa_fp = os.path.join(chip_dir, filename)
    if os.path.exists(chip_qa_fp) and not overwrite:
        click.echo("file already exists, exiting")
        return
    pbar = tqdm(total=len(os.listdir(chip_dir)))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        fut_nodata_results = {executor.submit(nodata_batch, chunk, chip_dir): chunk for chunk in paths_in_chunks(chip_dir)}
        for fut in concurrent.futures.as_completed(fut_nodata_results):
            path_chunk = fut_nodata_results[fut]
            scores = fut.result()
            for chip_fn, score in zip(path_chunk, scores):
                nodata_scorebook[score].append(chip_fn)
            pbar.update(len(scores))

    with open(chip_qa_fp, "w") as f:
        json.dump(nodata_scorebook, f)

    pbar.close()
    return chip_qa_fp




if __name__ == "__main__":
#    ard_paths = get_cog_paths(33)
    outfile = index_nodata()
    click.echo(outfile)


#    with Pool(10) as p:
#        print(p.map(write_chips, ard_paths))




