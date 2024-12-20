import requests
import pandas as pd
import os
from pathlib import Path
import json
from urllib.parse import quote, unquote
import yaml
import logging


def get_config(
    config_path: str | None = None
) -> dict:
    if not config_path:
        config_path = Path(__file__).parent.parent / \
            'config' / 'credentials.yml'

    if not os.path.isfile(config_path):
        raise ValueError('Missing config file or path incorrect.')

    if os.path.splitext(config_path)[-1] == '.json':
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
    elif os.path.splitext(config_path)[-1] in ['.yml', '.yaml']:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
    else:
        raise ValueError('Function not configured for non json/yaml files')

    return config


def get_logger(
        logger_name: str = 'ingestion',
        debug: bool = False):
    # Ensure the logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Create a custom logger
    logger = logging.getLogger(logger_name)

    # Set the default logging level
    logger.setLevel(logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('logs/ingestion.log')

    # Create formatters and add them to handlers
    c_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s [[%(filename)s:%(lineno)d]]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    f_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s [[%(filename)s:%(lineno)d]]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def get_stock_list(
    stock_list_path: str | None = None
) -> list:
    """ Stock tickers should be in the first column of the file """
    if not stock_list_path:
        stock_list_path = Path(__file__).parent.parent / \
            'data' / 'stock_list.csv'

    if not os.path.isfile(stock_list_path):
        raise ValueError('Missing stock list csv file or path incorrect.')

    if os.path.splitext(stock_list_path)[-1] != '.csv':
        raise ValueError('Function not configured for non csv files')

    return pd.read_csv(stock_list_path).iloc[:, 0].values.tolist()


def send_request(
    URL: str,
    username: str,
    password: str
) -> dict:
    base_url = 'https://www.comdinheiro.com.br/Clientes/API/EndPoint001.php'
    querystring = {'code': 'import_data'}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    output_format = 'json3'
    data = f'username={username}&'\
        f'password={password}&URL={URL}&'\
        f'format={output_format}'
    response = requests.post(
        base_url, data=data, headers=headers, params=querystring)
    response.raise_for_status()
    return response.json()


def get_fundamentalist_data(
    start_time: str,
    end_time: str,
    ticker: str,
    username: str,
    password: str,
    **kwargs
) -> dict:
    request_time_format = '%d/%m/%Y'
    start_time = format_request_time(start_time, request_time_format)
    end_time = format_request_time(end_time, request_time_format)
    URL = 'HistoricoIndicadoresFundamentalistas001.php?'\
        f'&data_ini={start_time}'\
        f'&data_fim={end_time}'\
        '&trailing=3&conv=MIXED' \
        '&moeda=MOEDA_ORIGINAL&c_c=consolidado'\
        '&m_m=1000000&n_c=2&f_v=1' \
        f'&papel={ticker}'\
        '&indic=NOME_EMPRESA+LL+RL+IPL+ROE+AT+PL+VPA+QUANT_ON'\
        '+QUANT_PN+QUANT_ON_PN+CCL+LG+DESPESA_FINANCEIRA'\
        '+DIVIDA_BRUTA+FCO+EBITDA+DEPRE_AMOR' \
        '&periodicidade=tri'\
        '&graf_tab=tabela&desloc_data_analise=1'\
        '&flag_transpor=0&c_d=d'\
        '&enviar_email=0&enviar_email_log=0'\
        '&cabecalho_excel=modo1'\
        '&relat_alias_automatico=cmd_alias_01'
    URL = quote(URL)
    return send_request(URL, username, password)


def get_market_data(
    start_time: str,
    end_time: str,
    ticker: str,
    username: str,
    password: str,
    flag_ajusted: int = 1,
    page: int = 1,
    **kwargs
) -> dict:
    request_time_format = '%d%m%Y'
    start_time = format_request_time(start_time, request_time_format)
    end_time = format_request_time(end_time, request_time_format)
    URL = 'HistoricoCotacaoAcao001-'\
        f'{ticker}-{start_time}-{end_time}-'\
        f'{flag_ajusted}-{page}'
    return send_request(URL, username, password)


def format_request_time(time_string: str, output_format: str) -> str:
    return pd.to_datetime(time_string).strftime(output_format)


if __name__ == '__main__':
    pass
    # # example usage
    # config = get_config('ingestion_scripts/secrets.json')
    # stock_list = get_stock_list('data/stock_list.csv')
    # end_time = pd.Timestamp.now()
    # start_time = end_time - pd.Timedelta(days=30)
    # start_time, end_time = start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d')
    # response = get_fundamentalist_data(start_time=start_time, end_time=end_time, ticker='PETR4', username=config['username'],password=config['password'])
