import requests

from colabfit_mcp.config import COLABFIT_AUTH, COLABFIT_BASE_URL


def search_datasets(
    text: str = None,
    elements: list[str] = None,
    property_types: list[str] = None,
    software: list[str] = None,
    page: int = 1,
    page_size: int = 10,
    advanced_filters: dict = None,
) -> dict:
    """Search ColabFit database for materials science datasets.

    Use this to find datasets for training interatomic potentials. Returns
    datasets with metadata including element coverage, configuration counts,
    and available properties (energy, forces, stress).

    Args:
        text: Search name, authors, and description fields.
        elements: Chemical elements to filter by (e.g. ["Si", "O"]).
        property_types: Filter by properties like "energy",
            "atomic_forces", "cauchy_stress".
        software: Filter by computation software (e.g. ["VASP"]).
        page: Page number for pagination (default 1).
        page_size: Results per page (default 10).
        advanced_filters: Additional query filters passed directly to API.

    Returns:
        Dict with results, pagination info, and next_step guidance.
    """
    query = {}
    if text:
        query["description"] = text
    if elements:
        query["elements"] = elements
    if property_types:
        query["property_types"] = property_types
    if software:
        query["software"] = software
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
        # Pagination is applied client-side; the API always returns the full result set.
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
                "Use download_dataset with a dataset_id to download "
                "training data, then fine_tune_mace to train a model."
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
