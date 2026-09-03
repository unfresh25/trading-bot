# Hybrid Trader V0

Baseline cuantitativo experimental para BTC/USD y SPY. Solo admite investigación,
backtesting y, en una fase posterior, paper trading. No incluye ejecución live ni
credenciales de broker.

## Decisiones de diseño

- La señal se calcula al cierre de la vela `t`; la entrada más temprana es `open[t+1]`.
- El target usa tres clases (`DOWN=-1`, `NEUTRAL=0`, `UP=1`) y cuatro velas de horizonte.
- Su umbral es el máximo entre un mínimo, volatilidad histórica escalada y costos de ida
  y vuelta multiplicados por un margen de seguridad.
- Walk-forward: 6 meses de entrenamiento, 1 de calibración reservado y 1 de prueba,
  avanzando un mes. La calibración se reserva pero todavía no se usa en el baseline.
- Estrategia long-only, capital independiente de USD 500 y riesgo inicial de 0,5%.
- Persistencia prevista: Parquet para datos y DuckDB para decisiones auditables.

## Instalación

```powershell
python -m pip install uv
python -m uv sync --extra dev --extra server
python -m uv run pytest
```

`uv` descarga Python 3.12 de acuerdo con `.python-version`, crea `.venv` y genera
`uv.lock`. Así PyArrow utiliza un wheel publicado y no intenta compilarse localmente.

## Datos y ejecución

El CSV debe contener `timestamp,open,high,low,close,volume`, con barras cerradas y ordenadas.
Se recomienda UTC. Se requieren al menos ocho meses completos después del calentamiento de
features.

```powershell
hybrid-trader .\data\processed\SPY_15m.csv
```

## API REST

La API es un adaptador versionado sobre el mismo caso de uso. No expone operaciones de
broker. Para iniciarla:

```powershell
python -m uv run uvicorn hybrid_trader.api:app --reload
```

OpenAPI y Swagger quedan disponibles en `http://127.0.0.1:8000/docs`.

| Método | Recurso | Resultado |
|---|---|---|
| `GET` | `/api/v1/health` | Estado del servicio y confirmación de que live está desactivado |
| `POST` | `/api/v1/backtests` | Crea un backtest síncrono desde barras OHLCV y devuelve su resumen |

Los errores de entrada usan una estructura estable con `error`, `message`, `details`,
`timestamp` y `path`. Los DTO no aceptan campos desconocidos, el símbolo tiene formato
restringido y cada vela valida timezone, precios, rango y volumen.

La API depende del protocolo `BacktestRunner`. `WalkForwardBacktestRunner` adapta el motor
actual, mientras que pruebas u otras implementaciones pueden sustituirlo desde
`create_app()` sin cambiar la presentación HTTP.

Los parámetros están en `src/hybrid_trader/config.py`. El motor registra sus predicciones
y trades en el resultado; `DuckDBDecisionStore` ofrece el contrato inicial de auditoría.

## Límites actuales

V0 aún no descarga datos, no calibra probabilidades, no simula stops intrabar ni fills
parciales y no incorpora XGBoost. Estas extensiones se deben comparar mediante pruebas de
ablación después de establecer resultados fiables del baseline logístico.
