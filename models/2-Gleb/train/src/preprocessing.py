import os
from utils import jread, jdump, json_record_to_poly, get_cortex_polygons
from pathlib import Path


def filt_glom_by_cortex(records_path_in: str, anot_struct_path_in: str, records_path_out: str) -> None:
    """ Filter glomeruli by location in Cortex. Load -> Filter -> Save.
    """

    assert records_path_in != records_path_out, "Please change paths to avoid overwriting input files"

    records_json = jread(Path(records_path_in))
    anot_structs_json = jread(Path(anot_struct_path_in))

    # Get list of Cortex polygons
    cortex_polygons = get_cortex_polygons(anot_structs_json)
    assert len(cortex_polygons) != 0, "No Cortex"

    filt_records_json = []
    for record in records_json:
        polygons = json_record_to_poly(record)
        # If at least one polygon intersects with at least one Сortex polygon, then append to filt_records_json
        if any([polyg.intersects(cortex_polygon) for cortex_polygon in cortex_polygons for polyg in polygons]):
            filt_records_json.append(record)
    assert len(filt_records_json) != 0, "No intersections are found"
    print(f"{len(records_json) - len(filt_records_json)} glomerulus are removed")

    os.makedirs(Path(records_path_out).parent, exist_ok=True)
    jdump(filt_records_json, Path(records_path_out))
