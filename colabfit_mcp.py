from mcp.server.fastmcp import FastMCP
import requests
from pathlib import Path

mcp = FastMCP("query-mcp")

@mcp.tool()
def dataset_query(
    name: str = None,
    authors: str  = None,
    description: str = None,
    elements: list[str] = None,
    exact_elements: bool = False,
    doi: str = None,
    min_co: int = None,
    max_co: int = None,
    min_elements: int = None,
    max_elements: int = None,
    min_atoms: int = None,
    max_atoms: int = None,
    property_types: list[str] = None,
    license: list[str] = None,
    equilibrium: bool = None,
    given_sort_by: str = None,
    given_sort_direction: str = "descending",
    software: list[str] = None,
    methods_text_filter: list[str] = None,
    page: int = 1,
    page_size: int = 10
):
    """
    Queries the ColabFit material and chemical database for datasets using a variety of arguments.
    These are:

    name: dataset name
    authors: dataset authors
    description: dataset description
    elements: list of chemical elements present in dataset
    exact_elements: whether of not to return datasets exactly matching elements. If False,
        if one queries for 'C', for example, then datasets with 'C' and other elements will be returned,
        otherwise datasets with only 'C' are returned.
    doi: str of the dataset doi
    min_co: minimum number of configurations in a dataset
    max_co: maximum number of configurations in a dataset
    min_elements: minimum number of distinct elements in the dataset
    max_elements: maximum number of distinct elements in the dataset
    min_atoms: minimum number of atoms in a dataset
    max_atoms: maximum number of atoms in a dataset
    property_types: property types present in the datasets. Can be
        'adsorption_energy', 'atomic_forces', 'atomization_energy', 'cauchy_stress',
        'electronic_band_gap', 'energy', 'energy_above_hull', and/or 'formation_energy'.
    license: license under which the dataset was released. Options are:
        APACHE-2.0
        BSD-3-CLAUSE
        CC-BY-3.0
        CC-BY-4.0
        CC-BY-NC-ND-4.0
        CC-BY-SA-4.0
        CC0
        CC0-1.0
        GPL-2.0
        GPL-2.0-ONLY
        GPL-3.0
        GPL-3.0-ONLY
        LGPL-3.0
        LGPL-3.0-ONLY
        MIT
        NIST-PD
    equilibrium: Whether or not datasets should be restricted to those that just contain equilibrium
        structures. Otherwise they contain nonequilibrium structures from relaxation or molecular
        dynamics trajectories, or similar.
    given_sort_by: options to sort returned datasets by:
        "nconfigurations": number of configurations
        "nelements": number of elements
        "nsites": number of atoms
        id": ColabFit ID
        "name": Name of dataset
        "downloads": Number of downloads
        "date_added_to_colabfit": Date added to colabFit
    given_sort_direction: Order returned results by in 'given_sort_by' in 'descending' or 'ascending' order
    software: software used to compute data, e.g Gaussian, VASP, etc.
    methods_text_filter: computational method, e.g., DFT-PBE, CCSD, etc.
    
    page: page number to return
    page_size: number of results to return in a page. This is useful to paginate long result responses

    This function returns a dictionary with keys:
        "Success" which is True if the query successfully ran
        "Results" which contain a list of results where each result is a dictionary containing info about the dataset returned or the error 
           if "Success" is False
        "Result Length" gives the length of the (possibly paginated) results
        "Page", "Page Size", "Total Pages" to help with pagination if query is successful
    """

    args_dict = locals()
    args_dict.pop('page')
    args_dict.pop('page_size')
    start = (page - 1) * page_size
    end = start + page_size
    try:
        response = requests.post("https://materials.colabfit.org/mcp/dataset-query", json=args_dict, auth=('mcp-tool', 'mcp-secret'))
        response_json = response.json()
        total_pages = (len(response_json) + page_size - 1) // page_size
        result = {"Success": True, "Results": response_json[start:end], "Result Length": len(response_json[start:end]), "Page": page, "Page Size": page_size, "Total Pages": total_pages}
    except Exception as e:
        result = {"Success": False, "Results": e} 
    return result

@mcp.tool()
def download_dataset(dataset_id: str = None, format: str = "parquet"):
    """
    Downloads a ColabFit dataset file in a variety of formats.
    Those are:
        "parquet" which represents the dataset post ingestion into the database
        "original" which is the original raw data file(s) 

    "dataset_id" is the associated ColabFit ID of the dataset which follows pattern DS_123456abcdef_0
    The file will be saved to <dataset_id>.tar,gz if parquet, otherwise <dataset_id>
    """
    download_dir = str(Path.home() / "Downloads")
    if format == "parquet":
        filename = f"{dataset_id}.tar.gz"
        saved_file = f"{download_dir}/{filename}"
    else:
        filename = f"{dataset_id}"
        saved_file = f"{download_dir}/{filename}.tar.xz"
    if dataset_id:
        try:
            url = f'https://materials.colabfit.org/mcp/dataset-download/{format}/{filename}' 
            with requests.get(url, stream=True, auth=('mcp-tool', 'mcp-secret')) as r:
                r.raise_for_status()
                with open(saved_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=10_000_000):
                        f.write(chunk)
            return {"Success": True, "Downloaded File": saved_file}
        except Exception as e:
            return {"Success": False, "Error": e}
