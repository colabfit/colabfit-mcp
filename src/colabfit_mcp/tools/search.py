import requests

from colabfit_mcp.config import COLABFIT_AUTH, COLABFIT_BASE_URL


def search_datasets(
    text: str = None,
    elements: list[str] = None,
    property_types: list[str] = None,
    software: list[str] = None,
    min_configurations: int = None,
    max_configurations: int = None,
    exact_elements: bool = False,
    sort_by: str = "downloads",
    sort_direction: str = "descending",
    page: int = 1,
    page_size: int = 10,
    advanced_filters: dict = None,
) -> dict:
    """Search ColabFit database for materials science datasets.

    ## DISCOVERY PROTOCOL — ALWAYS FOLLOW THIS ORDER

    1. Call check_local_datasets first to find already-downloaded datasets in the
       Docker container (/home/mcpuser/colabfit/datasets/). If a suitable match
       exists locally, skip steps 2-3 and proceed directly to train_mace.
    2. If no local match, call this tool to search ColabFit.org. ColabFit is the
       authoritative source — do NOT search the internet or guess dataset names.
    3. Download the chosen dataset with download_dataset, then train with train_mace.

    ## FILTERING BY DATASET SIZE — min_configurations / max_configurations

    These parameters filter by the ACTUAL nconfigurations count stored in the
    ColabFit database. They work as server-side filters — only datasets whose
    nconfigurations falls within [min_configurations, max_configurations] are
    returned. They do NOT change dataset content in any way.

    - To find small datasets (e.g. 100–500 configs): set max_configurations=500
    - To find datasets above a minimum size: set min_configurations=1000
    - To find datasets in a specific range: set both

    CRITICAL: These parameters have NO effect on download_dataset. A dataset with
    200 configurations will be downloaded with all 200 configurations — there is
    no mechanism to download a subset of a ColabFit dataset. The n_configs
    parameter in download_dataset does NOT reduce the downloaded data; see that
    tool's documentation for details.

    ## SORTING

    Use sort_by + sort_direction to control result ordering:
    - sort_by="nconfigurations", sort_direction="ascending" — smallest datasets first
    - sort_by="nconfigurations", sort_direction="descending" — largest datasets first
    - sort_by="downloads" (default) — most popular datasets first
    - sort_by="date_added_to_colabfit" — newest datasets first

    Valid sort_by values: "nconfigurations", "nelements", "nsites", "downloads",
    "date_added_to_colabfit", "name", "id".

    ## ELEMENT FILTERING

    - elements=["Si"] — datasets that contain Si (and possibly other elements)
    - elements=["Si"], exact_elements=True — datasets with ONLY Si
    - elements=["Si", "O"] — datasets that contain both Si and O

    ## SEARCHING BY TEXT

    The text parameter matches against dataset name AND description fields.
    All words in the text string must appear (AND logic). Example:
      text="silicon bulk crystal" returns datasets whose name or description
      contains all three words "silicon", "bulk", and "crystal".

    ## WORKFLOW NOTES

    - The API always requires energy AND atomic_forces; property_types adds
      additional requirements on top of this minimum.
    - Pagination is client-side: all matching results are fetched from the API
      then sliced. total_results reflects the full matching count.
    - Pass the 'name' field from results to download_dataset as dataset_name.
    - Pass the 'id' field from results to download_dataset as dataset_id.

    Args:
        text: Search dataset name and description fields (all words must match).
        elements: Chemical elements the dataset must contain (e.g. ["Si", "O"]).
        property_types: Additional required properties beyond energy + forces,
            e.g. ["cauchy_stress"].
        software: Filter by computation software (e.g. ["VASP"]).
        min_configurations: Only return datasets with nconfigurations >= this value.
            Does NOT affect what gets downloaded — only filters search results.
        max_configurations: Only return datasets with nconfigurations <= this value.
            Does NOT affect what gets downloaded — only filters search results.
            Use this to find small datasets (e.g. max_configurations=500).
        exact_elements: If True, only datasets whose element set exactly matches
            the elements list are returned. Default False (subset match).
        sort_by: Field to sort results by. Default "downloads". Options:
            "nconfigurations", "nelements", "nsites", "downloads",
            "date_added_to_colabfit", "name", "id".
        sort_direction: "ascending" or "descending". Default "descending".
        page: Page number for pagination (default 1).
        page_size: Results per page (default 10).
        advanced_filters: Additional raw query filters passed directly to the API.

    Returns:
        Dict with results list, pagination info, and next_step guidance.
        Each result includes:
          - 'name': pass to download_dataset as dataset_name
          - 'id': pass to download_dataset as dataset_id
          - 'nconfigurations': actual number of configurations in the dataset
          - 'elements': list of elements present
          - 'description': dataset description
    """
    required_properties = {"energy", "atomic_forces"}
    merged_properties = required_properties | set(property_types or [])
    query = {
        "property_types": sorted(merged_properties),
        "given_sort_by": sort_by,
        "given_sort_direction": sort_direction,
        "exact_elements": exact_elements,
    }
    if text:
        query["description"] = text
    if elements:
        query["elements"] = elements
    if software:
        query["software"] = software
    if min_configurations is not None:
        query["min_co"] = min_configurations
    if max_configurations is not None:
        query["max_co"] = max_configurations
    if advanced_filters:
        query.update(advanced_filters)

    start = (page - 1) * page_size
    end = start + page_size

    try:
        response = requests.post(
            f"{COLABFIT_BASE_URL}/mcp/dataset-query",
            json=query,
            auth=COLABFIT_AUTH,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        total = len(data)
        total_pages = (total + page_size - 1) // page_size
        results = data[start:end]

        return {
            "success": True,
            "results": results,
            "count": len(results),
            "page": page,
            "total_pages": total_pages,
            "total_results": total,
            "pagination_note": "Results are sliced client-side from the full API response.",
            "next_step": (
                "Use download_dataset with dataset_name (the 'name' field) "
                "and dataset_id (the 'id' field) to download training data, "
                "then train_mace to train a model."
            ),
        }
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"HTTP error: {e}"}
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot reach ColabFit API. Check network connectivity.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
