from multiprocessing import Pool
import concurrent.futures
import collections
import os
import re
import ujson as json
import vaex
import csv

def quadkey_regex(zoom=12):
    return re.compile(f"""'^\d{zoom}$'""")

def load_nodata_scores():
    with open("/home/ubuntu/data/chips/33/nodata_index.json", "r") as f:
        ndix = json.load(f)
    return ndix

qkp = quadkey_regex()
nd = load_nodata_scores()["all_data"]
qk_img_map = collections.defaultdict(list)
for n in nd:
    qk_img_map[n[4:21]].append(n)
qks = [qk.name for qk in os.scandir("/home/ubuntu/data/ard/33") if qkp.match(qk.name)]
with open("/home/ubuntu/data/labels.json") as r:
    labels = json.load(r)

labeling = sorted(labels.keys())

def write_qk_training_data(qk):
    print("do we get here")
    qklbls = vaex.open(f'''/home/ubuntu/data/ard/33/{qk}/qk_label_record.csv''', dtype=str)
    with open(f'''/home/ubuntu/data/ard/33/{qk}/training_data.csv''', "w") as td:
        writer = csv.writer(td)
        headers = ["image"] + list(labels.keys())
        writer.writerow(headers)
        n = 0
        for tileqk in qklbls.TileQuadkey.unique():
                #print(tileqk)
            if tileqk not in qk_img_map.keys():
                print("not in there")
                continue
            encoding = {lbl: 0 for lbl in labeling}
            qkdf = qklbls[qklbls.TileQuadkey == tileqk]
            for label in labeling:
                tagd = labels[label]
                if hit_label(qkdf, tagd):
                    encoding[label] = 1
                if 1 in encoding.values():
                    for img in qk_img_map[tileqk]:
                        row = [img] + list(encoding.values())
                        writer.writerow(row)
                        n += 1
                    #print(f'''wrote {tileqk}, n={nn}''')
    return n


def hit_label(qkdf, tagd):
    for key, vals in tagd.items():
        kf = qkdf.PrimaryKey.str.contains(key)
        if kf.values.any():
            tf = qkdf[kf]
            for val in vals:
                vf = tf.KeyValue.str.contains(val)
                if vf.values.any():
                    return True
    return False


def main():
    pass
#    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
#        print("in main!")
#        for n in executor.map(write_qk_training_data, qks):
#            print(n)

if __name__ == "__main__":
    with Pool(4) as p:
        print(p.map(write_qk_training_data, qks))
    main()
