from pydantic import BaseModel, Field
from datetime import datetime


class BaseMetadata(BaseModel):
    date_time: datetime
    feature: str

class RequestMarketMetadata(BaseModel):
    ticker: str = Field(alias='x')
    start_time: str = Field(alias='data_ini')
    end_time: str = Field(alias='data_fim')
    flag_ajusted: int = Field(alias='flag_ajusta')
    page: int = Field(alias='pagina')

class MarketMetadata(BaseMetadata):
    variables: dict[str, RequestMarketMetadata]


if __name__ == '__main__':
    # Example of usage
    metadata = {'date_time': '2024-10-07 19:40:11', 'user': 'username', 'server': 'minerva.comdinheiro.com.br', 'feature': 'HistoricoCotacaoAcao001', 'execution_time': '0.139', 'encode': 'utf8', 'JSON_UNESCAPED_UNICODE': True, 'JSON_PRETTY_PRINT': True, 'JSON_UNESCAPED_SLASHES': True, 'variables': {'get': {'x': 'C2PT34', 'data_ini': '01092024', 'data_fim': '01102024', 'flag_ajusta': '1', 'pagina': '1', 'flag_export': 'json3', 'ep': '1', 'ip_cliente': '000.000.000.00'}}}
    metadata_obj = BaseMetadata(**metadata)
    print(metadata_obj)
    market_meta_obj = MarketMetadata(**metadata)
    # market_meta_obj.variables["get"].ticker == 'C2PT34'