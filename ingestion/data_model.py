from pydantic import BaseModel, Field, field_validator
from datetime import datetime, date


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
