from pydantic import BaseModel, Field, field_validator
from datetime import datetime, date
import pandas as pd
from unidecode import unidecode


class BaseMetadata(BaseModel):
    date_time: datetime
    feature: str


class RequestMarketMetadata(BaseModel):
    ticker: str = Field(alias='x')
    start_time: date = Field(alias='data_ini')
    end_time: date = Field(alias='data_fim')
    flag_ajusted: int = Field(alias='flag_ajusta')
    page: int = Field(alias='pagina')

    @field_validator('start_time', mode='before')
    def format_start_time(cls, v):
        if isinstance(v, str):
            # If v is a string, assume it's in the format "ddmmyyyy"
            return date(int(v[4:8]), int(v[2:4]), int(v[:2]))
        return v

    @field_validator('end_time', mode='before')
    def format_end_time(cls, v):
        if isinstance(v, str):
            # If v is a string, assume it's in the format "ddmmyyyy"
            return date(int(v[4:8]), int(v[2:4]), int(v[:2]))
        return v


class MarketMetadata(BaseMetadata):
    variables: dict[str, RequestMarketMetadata]


class RequestFundamentalistMetadata(BaseModel):
    start_time: date = Field(alias='data_ini')
    end_time: date = Field(alias='data_fim')
    ticker: str = Field(alias='papel')
    trailing: int
    currency: str = Field(alias='moeda')
    m_m: int = 1000000
    parameters: list[str] = Field(alias='indic')
    frequency: str = Field(alias='periodicidade')

    @field_validator('parameters', mode='before')
    def split_parameters(cls, v):
        return v.split()  # Split by space

    @field_validator('start_time', mode='before')
    def format_start_time(cls, v):
        if isinstance(v, str):
            # If v is a string, assume it's in the format "dd/mm/yyyy"
            return date(int(v[6:10]), int(v[3:5]), int(v[:2]))
        return v

    @field_validator('end_time', mode='before')
    def format_end_time(cls, v):
        if isinstance(v, str):
            # If v is a string, assume it's in the format "dd/mm/yyyy"
            return date(int(v[6:10]), int(v[3:5]), int(v[:2]))
        return v


class FundamentalistMetadata(BaseMetadata):
    variables: dict[str, RequestFundamentalistMetadata]


class DataModel(BaseModel):
    tables: dict[str, dict]

    @field_validator('tables')
    def parse_table(cls, v):
        return {key: cls.parse_table_data(value) for key, value in v.items()}

    @staticmethod
    def parse_table_data(v):
        df = pd.DataFrame(v).T
        df = df.rename(columns=v['lin0'])
        df.columns = df.columns.str.replace('\n', ' ').str.strip().str.replace(' ', '_').str.lower(
        ).str.replace(r'r$', 'reais', regex=True).str.replace('(', '').str.replace(')', '')
        df.columns = [unidecode(col) for col in df.columns]
        df = df.drop(index='lin0')
        df = df.replace(',', '.', regex=True)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df = df.set_index('data')
        df = df.replace(['nd', 'DRE', 'DRN'], pd.NA)
        df = df.dropna(how='all', axis=1).dropna(how='all')
        return df

    class Config:
        arbitrary_types_allowed = True


if __name__ == '__main__':
    # Example of usage
    metadata = {'date_time': '2024-10-07 19:40:11', 'user': 'username', 'server': 'minerva.comdinheiro.com.br', 'feature': 'HistoricoCotacaoAcao001', 'execution_time': '0.139', 'encode': 'utf8', 'JSON_UNESCAPED_UNICODE': True,
                'JSON_PRETTY_PRINT': True, 'JSON_UNESCAPED_SLASHES': True, 'variables': {'get': {'x': 'C2PT34', 'data_ini': '01092024', 'data_fim': '01102024', 'flag_ajusta': '1', 'pagina': '1', 'flag_export': 'json3', 'ep': '1', 'ip_cliente': '000.000.000.00'}}}
    metadata_obj = BaseMetadata(**metadata)
    print(metadata_obj)
    market_meta_obj = MarketMetadata(**metadata)
    print(market_meta_obj)
    market_meta_obj.variables["get"].ticker == 'C2PT34'
    metadata = {'date_time': '2024-10-07 19:33:41', 'user': 'username', 'server': 'minerva.comdinheiro.com.br', 'feature': 'HistoricoIndicadoresFundamentalistas001', 'execution_time': '0.140', 'encode': 'utf8', 'JSON_UNESCAPED_UNICODE': True, 'JSON_PRETTY_PRINT': True, 'JSON_UNESCAPED_SLASHES': True, 'variables': {'get': {'ep': '1', 'data_ini': '01/09/2024', 'data_fim': '01/10/2024', 'trailing': '3', 'conv': 'MIXED', 'moeda': 'MOEDA_ORIGINAL', 'c_c': 'consolidado', 'm_m': '1000000',
                                                                                                                                                                                                                                                                                                                                    'n_c': '2', 'f_v': '1', 'papel': 'C2PT34', 'indic': 'NOME_EMPRESA LL RL IPL ROE AT PL VPA QUANT_ON QUANT_PN QUANT_ON_PN CCL LG DESPESA_FINANCEIRA DIVIDA_BRUTA FCO EBITDA DEPRE_AMOR', 'periodicidade': 'tri', 'graf_tab': 'tabela', 'desloc_data_analise': '1', 'flag_transpor': '0', 'c_d': 'd', 'enviar_email': '0', 'enviar_email_log': '0', 'cabecalho_excel': 'modo1', 'relat_alias_automatico': 'cmd_alias_01', 'flag_export': 'json3', 'ip_cliente': '000.000.000.00'}}}
    fundamentalist_meta_obj = FundamentalistMetadata(**metadata)
    print(fundamentalist_meta_obj)
    market_data = {'tables': {'tab0': {'lin0': {'col0': 'Data', 'col1': 'Fech Ajustado', 'col2': 'Variação(%)', 'col3': 'Fech Histórico', 'col4': 'Abertura Ajustado', 'col5': 'Mín Ajustado', 'col6': 'Médio Ajustado', 'col7': 'Máx Ajustado', 'col8': 'Vol (MM R$)', 'col9': 'Negócios', 'col10': 'Fator', 'col11': 'Tipo', 'col12': 'Quant em Aluguel', 'col13': 'Vol em Aluguel(MM R$)'}, 'lin1': {'col0': '02/09/2024', 'col1': 'nd', 'col2': 'nd', 'col3': 'nd', 'col4': 'nd', 'col5': 'nd', 'col6': 'nd', 'col7': 'nd', 'col8': 'nd', 'col9': 'nd', 'col10': 'nd', 'col11': 'nd', 'col12': 'nd', 'col13': 'nd'}, 'lin2': {
        'col0': '03/09/2024', 'col1': 'nd', 'col2': 'nd', 'col3': 'nd', 'col4': 'nd', 'col5': 'nd', 'col6': 'nd', 'col7': 'nd', 'col8': 'nd', 'col9': 'nd', 'col10': 'nd', 'col11': 'nd', 'col12': 'nd', 'col13': 'nd'}, 'lin3': {'col0': '04/09/2024', 'col1': '46,5376660921', 'col2': 'nd', 'col3': '46,8', 'col4': '46,5376660912', 'col5': '46,5376660912', 'col6': '46,5376660912', 'col7': '46,5376660912', 'col8': '0,000234', 'col9': '1,00', 'col10': '1', 'col11': 'DRN', 'col12': 'nd', 'col13': 'nd'}}}}
    market_data_obj = DataModel(**market_data)
    print(market_data_obj)
    fundamentalist_data = {'tables': {'tab0': {'lin0': {'col0': 'Data', 'col1': 'C2PT34\nNome da Empresa', 'col2': 'C2PT34\nLucro Líquido\n3 meses\nConsolidado\n Milhões\n', 'col3': 'C2PT34\nReceita Líquida\n3 meses\nConsolidado\n Milhões\n', 'col4': 'C2PT34\nP/L', 'col5': 'C2PT34\nROE', 'col6': 'C2PT34\nAtivo Total\nConsolidado\n Milhões\n', 'col7': 'C2PT34\nPatrimônio Líquido\nConsolidado\n Milhões\n', 'col8': 'C2PT34\nVPA', 'col9': 'C2PT34\nQUANT_ON', 'col10': 'C2PT34\nQUANT_PN', 'col11': 'C2PT34\nQUANT_ON_PN', 'col12': 'C2PT34\nCCL', 'col13': 'C2PT34\nLG', 'col14': 'C2PT34\nDespesa Financeira\n3 meses\nConsolidado\n Milhões\n', 'col15': 'C2PT34\nDívida Bruta\n Milhões', 'col16': 'C2PT34\nFluxo de Caixa das Operações\n3 meses\nConsolidado\n Milhões\n', 'col17': 'C2PT34\nEBITDA\n3 meses\nConsolidado\n Milhões\n',
                                                        'col18': 'C2PT34\nDespesa de Depreciação, Amortização e Exaustão\n3 meses\nConsolidado\n Milhões\n', 'col19': 'C2PT34\nConsolidado\nou\nControlador', 'col20': 'C2PT34\nConvenção', 'col21': 'C2PT34\nMoeda', 'col22': 'C2PT34\nData da Demonstração', 'col23': 'meses', 'col24': 'Data da Análise'}, 'lin1': {'col0': '30/06/2024', 'col1': 'CAMDEN PROPERTY TRUST', 'col2': '42,917', 'col3': '387,15', 'col4': '275,807981453', 'col5': '0,883895177368', 'col6': '9079,574', 'col7': '4855,44', 'col8': '44,7894028006', 'col9': '108,640497', 'col10': '0', 'col11': '108,640497', 'col12': '-304,651', 'col13': '0,0281674315018', 'col14': '32,227', 'col15': '3552,81', 'col16': '225,069', 'col17': '222,392', 'col18': '145,894', 'col19': 'consolidado', 'col20': 'US GAAP', 'col21': 'USD', 'col22': '30/06/2024', 'col23': '3', 'col24': '01/07/2024'}}}}
    fundamentalist_data_obj = DataModel(**fundamentalist_data)
    print(fundamentalist_data_obj)
