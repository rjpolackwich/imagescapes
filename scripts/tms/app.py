
"""rio-tiler tile server."""

import os
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import uvicorn
from fastapi import FastAPI, Path, Query
from rasterio.crs import CRS
from starlette.background import BackgroundTask
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import Response

from rio_tiler.profiles import img_profiles
from rio_tiler.utils import render
from rio_tiler.io import COGReader
from cogeo_mosaic.mosaic import MosaicJSON
from cogeo_mosaic.backends import MosaicBackend
from cogeo_mosaic.errors import NoAssetFoundError


# From developmentseed/titiler
drivers = dict(jpg="JPEG", png="PNG")
mimetype = dict(png="image/png", jpg="image/jpg",)

class ImageType(str, Enum):
    """Image Type Enums."""

    png = "png"
    jpg = "jpg"



class TileResponse(Response):
    """Tiler's response."""

    def __init__(
        self,
        content: bytes,
        media_type: str,
        status_code: int = 200,
        headers: dict = {},
        background: BackgroundTask = None,
        ttl: int = 3600,
    ) -> None:
        """Init tiler response."""
        headers.update({"Content-Type": media_type})
        if ttl:
            headers.update({"Cache-Control": "max-age=3600"})
        self.body = self.render(content)
        self.status_code = 200
        self.media_type = media_type
        self.background = background
        self.init_headers(headers)


app = FastAPI(
    title="rio-tiler",
    description="A lightweight Cloud Optimized GeoTIFF tile server",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=0)

responses = {
    200: {
        "content": {"image/png": {}, "image/jpg": {}},
        "description": "Return an image.",
    }
}
tile_routes_params: Dict[str, Any] = dict(
    responses=responses, tags=["tiles"], response_class=TileResponse
)


mosaic = MosaicBackend("/home/ubuntu/data/ard/33/mosaic.json")

@app.get("/{z}/{x}/{y}", **tile_routes_params)
def tile(
    z: int,
    x: int,
    y: int,
    ):
    """Handle tiles requests."""
#    with COGReader(url) as cog:
#        tile, mask = cog.tile(x, y, z, tilesize=256)
    try:
        (tile, mask), assets = mosaic.tile(x, y, z)
    except NoAssetFoundError:
        pass #Do Something

    format = ImageType.jpg if mask.all() else ImageType.png

    driver = drivers[format.value]
    options = img_profiles.get(driver.lower(), {})
    img = render(tile, mask, img_format=driver, **options)

    return TileResponse(img, media_type=mimetype[format.value])


@app.get("/tilejson.json", responses={200: {"description": "Return a tilejson"}})
def tilejson(
    request: Request,
    url: str = Query(..., description="Cloud Optimized GeoTIFF URL."),
    minzoom: Optional[int] = Query(None, description="Overwrite default minzoom."),
    maxzoom: Optional[int] = Query(None, description="Overwrite default maxzoom."),
):
    """Return TileJSON document for a COG."""
    tile_url = request.url_for("tile", {"z": "{z}", "x": "{x}", "y": "{y}"}).replace("\\", "")

    kwargs = dict(request.query_params)
    kwargs.pop("tile_format", None)
    kwargs.pop("tile_scale", None)
    kwargs.pop("minzoom", None)
    kwargs.pop("maxzoom", None)

    qs = urlencode(list(kwargs.items()))
    tile_url = f"{tile_url}?{qs}"

    with COGReader(url) as cog:
        center = list(cog.center)
        if minzoom:
            center[-1] = minzoom
        tjson = {
            "bounds": cog.bounds,
            "center": tuple(center),
            "minzoom": minzoom or cog.minzoom,
            "maxzoom": maxzoom or cog.maxzoom,
            "name": os.path.basename(url),
            "tiles": [tile_url],
        }

    return tjson
