# Hybrid Trader V0

Baseline cuantitativo experimental para BTC/USD y SPY. Solo admite investigación,
backtesting y, en una fase posterior, paper trading. No incluye ejecución live ni
credenciales de broker.

## Decisiones de diseño

- La señal se calcula al cierre de la vela `t`; la entrada más temprana es `open[t+1]`.
- El target usa tres clases (`DOWN=-1`, `NEUTRAL=0`, `UP=1`) y una hora de horizonte
  para barras de 15 minutos. Entrada y salida deben existir en el timestamp exacto;
  nunca se comprime el tiempo saltando una barra ausente.
- Su umbral es el máximo entre un mínimo, volatilidad histórica escalada y costos de ida
  y vuelta multiplicados por un margen de seguridad.
- Walk-forward: 6 meses de entrenamiento, 1 de calibración y 1 de prueba, avanzando un
  mes. Aplica purge de 75 minutos y embargo de 15 minutos. El mes de calibración compara
  identidad, sigmoid e isotonic por log loss; el test solo recibe el método elegido.
- Estrategia long-only, capital independiente de USD 500 y riesgo inicial de 0,5%.
- Persistencia: Parquet para datos, predicciones, decisiones y trades; cada ejecución
  genera un manifiesto con filas y SHA-256. DuckDB sigue disponible para eventos online.

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

## Adquisición y versionado de datos

Las credenciales de market data se leen exclusivamente del entorno:

```powershell
$env:APCA_API_KEY_ID = "..."
$env:APCA_API_SECRET_KEY = "..."
```

Descarga sugerida para los experimentos iniciales:

```powershell
python -m uv run hybrid-data alpaca SPY stock --start 2021-09-01T00:00:00+00:00 --end 2026-09-01T00:00:00+00:00
python -m uv run hybrid-data alpaca BTC/USD crypto --start 2023-09-01T00:00:00+00:00 --end 2026-09-01T00:00:00+00:00
```

También se puede incorporar un CSV local:

```powershell
python -m uv run hybrid-data csv .\input.csv SPY stock
```

Cada ejecución crea una carpeta inmutable con `bars.parquet` y `metadata.json`. Los
metadatos contienen símbolo, proveedor, timeframe, rango, filas, fecha de creación y
SHA-256. La lectura verifica el hash antes de aceptar el dataset.

El diagnóstico de calidad estima barras faltantes para cripto 24/7. Para acciones reporta
gaps observados sin marcarlos automáticamente como faltantes, porque debe incorporarse el
calendario bursátil antes de distinguir cierres normales de huecos reales.

Para comparar datasets versionados:

```powershell
python -m uv run hybrid-report --dataset SPY=.\spy\bars.parquet --dataset BTC/USD=.\btc-usd\bars.parquet
```

El reporte separa por instrumento la estrategia walk-forward, buy & hold, momentum sobre
SMA-20 y mean reversion sobre z-score de 20 velas. Todos los benchmarks son long-only,
entran en la vela siguiente e incluyen comisión, spread y slippage.

### Controles experimentales por mercado

- SPY se filtra con el calendario `XNYS` a sesión regular, respetando festivos y cierres
  tempranos. Las barras premarket y after-hours no entran al experimento.
- BTC/USD conserva su reloj 24/7. Las barras con volumen cero se excluyen del entrenamiento
  y activan el veto estable `zero_volume` durante la simulación.
- SPY usa 0 bps de comisión, 1 bp de spread y 1 bp de slippage por lado.
- BTC/USD usa 0 bps de comisión, 8 bps de spread y 5 bps de slippage por lado.
- No se elimina físicamente ninguna barra de BTC: hacerlo comprimiría el reloj y podría
  convertir una hora real en un horizonte distinto. El volumen cero solo se excluye del
  entrenamiento y veta la ejecución.

### Diagnósticos y auditoría

Cada predicción fuera de muestra genera exactamente una decisión `BUY` o `NO_TRADE` con
un código estable: confianza baja, edge insuficiente, volumen cero, posición abierta,
cooldown, ATR inválido o límite de drawdown. `BacktestResult.decisions` conserva además
probabilidades, edge estimado, volumen, fold y régimen.

Los reportes incorporan:

- Distribución `DOWN`, `NEUTRAL` y `UP`.
- Brier score para `UP`, log loss multiclase y baselines ingenuos.
- Brier skill score y curva de calibración por bins.
- Resultados predictivos por fold, año y régimen.
- Resultados de trades por fold, año y régimen.
- Sharpe, Sortino, turnover y máxima racha de pérdidas.
- Conteo de todos los motivos `NO_TRADE`.
- Método de calibración y log loss de los tres candidatos por fold.
- Intervalo de confianza del retorno neto mediante block bootstrap de retornos diarios.

`hybrid-report` escribe por defecto `artifacts/research-report.json`; puede fijarse otra
ruta con `--output`. Junto al JSON crea `<nombre>-audit/<mercado>/` con
`predictions.parquet`, `decisions.parquet`, `trades.parquet` y `manifest.json`. Los
artefactos permanecen fuera de Git para evitar versionar datos o resultados voluminosos.

## Límites actuales

V0 aún no simula stops intrabar ni fills parciales y no incorpora XGBoost. La adquisición
actual usa Alpaca o CSV y el baseline logístico ya calibra probabilidades sin tocar el
periodo de prueba. XGBoost debe incorporarse después como una ablación sobre exactamente
los mismos folds bloqueados.
