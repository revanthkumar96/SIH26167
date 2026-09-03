"""Geospatial input handling: raster inspection, previews and compatibility checks."""

from satquery.geo.checks import (
    assign_roles,
    check_pair,
    infer_input_config,
    infer_modality,
)
from satquery.geo.raster import RasterInfo, read_bands, read_info, render_preview

__all__ = [
    "RasterInfo",
    "assign_roles",
    "check_pair",
    "infer_input_config",
    "infer_modality",
    "read_bands",
    "read_info",
    "render_preview",
]
