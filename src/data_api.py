"""
Open data downloader for Brazilian public institutions.

Supports four sources:
    - ANA   (Agência Nacional de Águas) via SOAP web service
    - ONS   (Operador Nacional do Sistema Elétrico) via CKAN
    - ANEEL (Agência Nacional de Energia Elétrica) via CKAN
    - CCEE  (Câmara de Comercialização de Energia Elétrica) via CKAN

Only the ONS path is used in this study. The others are kept for
completeness because they share infrastructure.
"""

import calendar
import io
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool

import pandas as pd
import requests
from tqdm import tqdm


def resolve_host(institution):
    mapping = {
        "ccee": "https://dadosabertos.ccee.org.br",
        "ons": "https://dados.ons.org.br",
        "aneel": "https://dadosabertos.aneel.gov.br",
    }
    return mapping.get(institution.lower())


def list_products(institution):
    if institution == "ana":
        return ["vazao", "chuva", "cota"]

    host = resolve_host(institution)
    try:
        r = requests.get(f"{host}/api/3/action/package_list")
        r.raise_for_status()
        return r.json().get("result", [])
    except Exception as e:
        print(f"Error listing products: {e}")
        return []


def list_ana_stations(product=None, state="", city=""):
    """List ANA hydrological stations (streamflow, precipitation, or stage)."""
    if product is None:
        raise ValueError(
            "Product name is required ('vazao', 'chuva', or 'cota')"
        )
    if product == "vazao":
        station_type = "1"
    elif product == "chuva":
        station_type = "2"
    elif product == "cota":
        station_type = "3"

    url = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario"
    params = {
        "codEstDE": "",
        "codEstATE": "",
        "tpEst": station_type,
        "nmEst": "",
        "nmRio": "",
        "codSubBacia": "",
        "codBacia": "",
        "nmMunicipio": city,
        "nmEstado": state,
        "sgResp": "",
        "sgOper": "",
        "telemetrica": "",
    }

    response = requests.get(url, params=params, timeout=120)
    tree = ET.ElementTree(ET.fromstring(response.content))
    root = tree.getroot()

    stations = []
    for station in root.iter("Table"):
        stations.append(
            {
                "Code": f'{int(station.find("Codigo").text):08}',
                "Name": station.find("Nome").text,
                "City": station.find("nmMunicipio").text,
                "State": station.find("nmEstado").text,
                "Latitude": float(station.find("Latitude").text),
                "Longitude": float(station.find("Longitude").text),
            }
        )

    return pd.DataFrame(stations)


def ana_data(list_station, data_type, threads=10):
    """Download historical series from the ANA SOAP service."""
    if type(list_station) is not list:
        list_station = [list_station]
    data_types = {"3": ["Vazao{:02}"], "2": ["Chuva{:02}"], "1": ["Cota{:02}"]}

    def __call_request(station):
        params = {
            "codEstacao": str(station),
            "dataInicio": "",
            "dataFim": "",
            "tipoDados": data_type,
            "nivelConsistencia": "",
        }

        response = requests.get(
            "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica",
            params,
            timeout=120.0,
        )

        tree = ET.ElementTree(ET.fromstring(response.content))
        root = tree.getroot()

        df = []
        for month in root.iter("SerieHistorica"):
            code = month.find("EstacaoCodigo").text
            code = f"{int(code):08}"
            consist = int(month.find("NivelConsistencia").text)
            date = pd.to_datetime(month.find("DataHora").text, dayfirst=False)
            date = pd.Timestamp(date.year, date.month, 1, 0)
            last_day = calendar.monthrange(date.year, date.month)[1]
            month_dates = pd.date_range(date, periods=last_day, freq="D")
            data = []
            list_consist = []
            for i in range(last_day):
                value = data_types[params["tipoDados"]][0].format(i + 1)
                try:
                    data.append(float(month.find(value).text))
                    list_consist.append(consist)
                except TypeError:
                    data.append(month.find(value).text)
                    list_consist.append(consist)
                except AttributeError:
                    data.append(None)
                    list_consist.append(consist)
            index_multi = list(zip(month_dates, list_consist))
            index_multi = pd.MultiIndex.from_tuples(
                index_multi, names=["Date", "Consistence"]
            )
            df.append(pd.DataFrame({code: data}, index=index_multi))
        if len(df) == 0:
            return pd.DataFrame()
        df = pd.concat(df)
        df = df.sort_index()

        drop_index = df.reset_index(level=1, drop=True).index.duplicated(keep="last")
        df = df[~drop_index]
        df = df.reset_index(level=1, drop=True)

        series = df[code]
        date_index = pd.date_range(series.index[0], series.index[-1], freq="D")
        series = series.reindex(date_index)
        return series

    if len(list_station) < threads:
        threads = len(list_station)

    with ThreadPool(threads) as pool:
        responses = list(
            tqdm(pool.imap(__call_request, list_station), total=len(list_station))
        )
    responses = [response for response in responses if not response.empty]
    data_stations = pd.concat(responses, axis=1)
    date_index = pd.date_range(
        data_stations.index[0], data_stations.index[-1], freq="D"
    )
    data_stations = data_stations.reindex(date_index)

    first, last = (
        data_stations.first_valid_index(),
        data_stations.last_valid_index(),
    )
    data_stations = data_stations.loc[first:last]

    return data_stations


def list_product_files(host, product):
    """Find the URLs of all files inside a CKAN product."""
    r = requests.get(f"{host}/api/3/action/package_show?id={product}")
    data = r.json()
    if data.get("success"):
        return [
            {"url": item["url"], "format": item.get("format", "").lower()}
            for item in data["result"]["resources"]
        ]
    return []


def download_csv_file(url):
    """Download a single CSV file from the product."""
    try:
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        df = pd.read_csv(
            io.BytesIO(resp.content),
            sep=";",
            encoding="latin-1",
            on_bad_lines="skip",
        )
        return df
    except Exception:
        return pd.DataFrame()


def download_product_data(institution, product):
    """Download all CSV files for a product and concatenate them."""
    host = resolve_host(institution)
    resources = list_product_files(host, product)

    csv_links = [res["url"] for res in resources if "csv" in res["format"]]
    if not csv_links:
        print(f"Product '{product}' has no CSV files.")
        return None

    print(f"Starting download of {len(csv_links)} files...")
    dfs = [download_csv_file(url) for url in csv_links]
    valid_dfs = [df for df in dfs if not df.empty]

    if valid_dfs:
        return pd.concat(valid_dfs, ignore_index=True)
    return None


def collect_data(institution, product=None, stations=None):
    """
    Main entry point. Routes to the right downloader based on the
    institution name.
    """
    if institution.lower() == "ana":
        if stations is None or product is None:
            raise ValueError(
                "Product ('vazao', 'chuva', or 'cota') and stations are required"
            )
        if product == "vazao":
            kind = "3"
        elif product == "chuva":
            kind = "2"
        elif product == "cota":
            kind = "1"
        return ana_data(stations, kind)

    elif institution.lower() in ["ons", "aneel", "ccee"]:
        if product is None:
            raise ValueError("Product name is required")
        return download_product_data(institution, product)

    else:
        raise ValueError(
            "Invalid institution. Use 'ana', 'ons', 'aneel', or 'ccee'."
        )
