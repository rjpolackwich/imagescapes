import click
from collections import defaultdict
from maxar_canvas_grid import Cell
import rasterio
import ujson as json


def build_chip_mapping(mocha_json):
    click.echo("building chip mapping")
    chip_mapping = defaultdict(list)
    for image in mocha_json.get('images', []):
        quad_id, catalog_id = image['file_name'][:-5].split('_')
        l12_quad_id = quad_id[:16]
        chip_mapping[(l12_quad_id, catalog_id)].append(quad_id)

    return chip_mapping


def get_path(ard_prefix, quad_id, catalog_id):
    zone, quad_key = quad_id.split('-')
    zone = int(zone[1:])
    
    with open(f'{ard_prefix}/acquisition_collections/{catalog_id}_collection.json') as stac_fh:
        stac_json = json.load(stac_fh)
        
    date_str = stac_json['extent']['temporal']['interval'][0][0]
    date_out = date_str.split('T')[0]
    
    return f'{ard_prefix}/{zone}/{quad_key}/{date_out}/{catalog_id}-visual.tif'


def chip_tiles(chip_mapping, ard_prefix, chips_prefix):
    '''
    builds mappings (zoom 12 quad_id, catalog_id) -> list of zoom 18 quadkeys to chip
    '''
    total_tiles_count = len(chip_mapping)
    total_chips_count = sum([len(chips) for chips in chip_mapping.values()])

    chips_count, tiles_count = 0, 0
    for (l12_quad_id, catalog_id), quad_ids in chip_mapping.items():
        image_path = get_path(ard_prefix, l12_quad_id, catalog_id)
        with rasterio.open(image_path) as src:
            for quad_id in quad_ids:
                window = rasterio.windows.from_bounds(*Cell(quad_id).bounds, transform=src.transform)
                with rasterio.open(f'{chips_prefix}/{quad_id}_{catalog_id}.jpg', 'w', 
                                driver='jpeg', 
                                width=256, 
                                height=256, 
                                count=src.count,
                                dtype=src.profile['dtype']) as dst:
                    dst.write(src.read(window=window))
                chips_count += 1
            tiles_count += 1
            
            click.echo(f"{chips_count} chips of {total_chips_count}")
            click.echo(f"{tiles_count} tiles of {total_tiles_count}")


@click.command()
@click.option('--mocha', type=str, required=True, help="path to mocha json")
@click.option("--ard-prefix", type=str, required=True, help="path to ard prefix")
@click.option("--chips-prefix", type=str, required=True, help="path to chips prefix (must exist)")
def main(mocha, ard_prefix, chips_prefix):
    with open(mocha) as fh:
        mocha_json = json.load(fh)

    chip_mapping = build_chip_mapping(mocha_json)
    chip_tiles(chip_mapping, ard_prefix, chips_prefix)


if __name__ == '__main__':
    main()
