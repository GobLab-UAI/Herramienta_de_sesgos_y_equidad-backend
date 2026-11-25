"""
eda.py — Análisis Exploratorio de Datos (EDA) orientado a sesgos.

Perfila un DataFrame y devuelve JSON (sin gráficos; el frontend los dibuja) para
evidenciar posibles **desbalances** y **sesgos** antes del análisis de equidad:

  - Perfil por columna: tipo, % de nulos, cardinalidad, top-valores, stats.
  - Matriz de asociación (Cramér's V) entre variables categóricas → detecta
    **proxies** (p. ej. "comuna" fuertemente asociada a "nivel socioeconómico").
  - Alertas de calidad: nulos altos, alta cardinalidad, grupos pequeños,
    categorías dominantes (desbalance), columnas constantes y posibles proxies.
"""
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Columnas "categórico-like" para asociación/distribución: pocas categorías.
MAX_CATEGORIES_FOR_ASSOC = 50
# Top-k categorías a devolver por columna (el resto se agrega en "Otros").
TOP_K = 12
# Categorías máximas por eje en las tablas de contingencia (bivariado).
MAX_CATEGORIES_FOR_CROSSTAB = 15
# Máximo de bins en histogramas de variables numéricas.
MAX_HIST_BINS = 20


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _column_dtype(series: pd.Series, n_unique: int) -> str:
    """Clasifica la columna en 'binary' | 'numeric' | 'categorical'."""
    if n_unique <= 2:
        return "binary"
    if _is_numeric(series) and n_unique > 10:
        return "numeric"
    return "categorical"


def _top_values(series: pd.Series, total: int) -> List[Dict]:
    counts = series.value_counts(dropna=True)
    top = counts.head(TOP_K)
    out = [
        {"value": str(idx), "count": int(cnt), "pct": round(100 * cnt / total, 2)}
        for idx, cnt in top.items()
    ]
    remaining = int(counts.iloc[TOP_K:].sum()) if len(counts) > TOP_K else 0
    if remaining:
        out.append({"value": "Otros", "count": remaining,
                    "pct": round(100 * remaining / total, 2), "is_aggregate": True})
    return out


def _normalized_entropy(counts: np.ndarray) -> float:
    """Entropía de Shannon normalizada [0, 1]. 1 = perfectamente balanceada,
    cercano a 0 = una categoría concentra casi todo."""
    total = counts.sum()
    if total <= 0 or len(counts) <= 1:
        return 1.0
    p = counts / total
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    hmax = math.log(len(counts))
    return float(h / hmax) if hmax > 0 else 1.0


def _balance_label(evenness: float) -> str:
    if evenness >= 0.85:
        return "equilibrada"
    if evenness >= 0.6:
        return "moderada"
    return "desbalanceada"


def _histogram(series: pd.Series) -> List[Dict]:
    """Histograma de una variable numérica: lista de bins {x0, x1, count}."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return []
    n_bins = int(min(MAX_HIST_BINS, max(5, math.ceil(math.sqrt(len(values))))))
    counts, edges = np.histogram(values, bins=n_bins)
    return [
        {"x0": round(float(edges[i]), 4), "x1": round(float(edges[i + 1]), 4),
         "count": int(counts[i])}
        for i in range(len(counts))
    ]


_PRED_KEYS = ("pred", "score", "prob", "riesgo", "resultado", "yhat", "y_hat")
_ACTUAL_KEYS = ("real", "label", "target", "y_true", "ytrue", "actual", "verdad", "observ",
                "etiqueta", "outcome", "clase")

# Roles posibles: "id" (identificador), "prediction" (predicción del modelo),
# "actual" (valor real), "protected" (variable protegida) y "unused" (sin uso).
# "outcome" queda solo cuando hay predicción/valor real indistinguibles por nombre.


def _is_id(n_unique: int, n_rows: int) -> bool:
    return bool(n_rows) and n_unique / n_rows > 0.5 and n_unique > 20


def _value_set(series: pd.Series) -> frozenset:
    return frozenset(str(v) for v in series.dropna().unique())


def _assign_roles(df: pd.DataFrame, columns: List[Dict], n_rows: int) -> None:
    """Asigna ``role_hint`` a cada columna (heurística; el usuario decide).

    1. Identificador: casi un valor distinto por fila.
    2. Predicción / valor real: por nombre de la columna. Si solo se reconoce una,
       la otra es la columna (no protegida) con el mismo conjunto de valores.
    3. Variable protegida: categórica, o binaria de texto (p. ej. sexo).
    4. Sin uso: el resto (p. ej. numéricas continuas).
    """
    by_name = {c["name"]: c for c in columns}
    for c in columns:
        c["role_hint"] = "id" if _is_id(c["unique"], n_rows) else None

    def free(c):
        return c["role_hint"] is None and 2 <= c["unique"] <= MAX_CATEGORIES_FOR_ASSOC

    def first_with(keys, exclude=None):
        for c in columns:
            if free(c) and c["name"] != exclude and any(k in c["name"].lower() for k in keys):
                return c["name"]
        return None

    pred = first_with(_PRED_KEYS)
    actual = first_with(_ACTUAL_KEYS, exclude=pred)
    if bool(pred) != bool(actual):
        known = pred or actual
        known_set = _value_set(df[known])
        for c in columns:
            if free(c) and c["name"] != known and _value_set(df[c["name"]]) == known_set:
                if pred:
                    actual = c["name"]
                else:
                    pred = c["name"]
                break
    if pred:
        by_name[pred]["role_hint"] = "prediction"
    if actual:
        by_name[actual]["role_hint"] = "actual"

    for c in columns:
        if c["role_hint"] is not None:
            continue
        numeric = _is_numeric(df[c["name"]])
        if c["dtype"] == "categorical" or (c["dtype"] == "binary" and not numeric):
            c["role_hint"] = "protected"
        elif c["dtype"] == "binary" and not (pred and actual):
            # Binaria numérica {0,1} sin nombre reconocible: predicción o valor real.
            c["role_hint"] = "outcome"
        else:
            c["role_hint"] = "unused"


def _fmt_int(n: int) -> str:
    """Entero con punto de miles (es-CL): 7000 -> '7.000'."""
    return f"{int(n):,}".replace(",", ".")


def _fmt_dec(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _join_es(items: List[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def _positive_value(series: pd.Series):
    """Valor positivo de la columna de valor real si es binaria {0,1}; si no, None."""
    vals = set(pd.to_numeric(series.dropna(), errors="coerce").dropna().unique())
    if vals and vals <= {0.0, 1.0} and series.dropna().nunique() <= 2:
        return 1.0
    return None


def _capped_categories(series: pd.Series, cap: int = MAX_CATEGORIES_FOR_CROSSTAB) -> List[str]:
    """Top-`cap` categorías por frecuencia (como str)."""
    return [str(v) for v in series.value_counts(dropna=True).head(cap).index]


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V corregido (Bergsma) entre dos series categóricas. Rango [0, 1]."""
    confusion = pd.crosstab(x, y)
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(min(1.0, math.sqrt(phi2corr / denom)))


def run_eda(df: pd.DataFrame, min_group_size: int = 50) -> Dict:
    """Genera el perfil exploratorio del DataFrame."""
    n_rows = int(len(df))
    n_cols = int(df.shape[1])
    total_cells = max(1, n_rows * n_cols)
    total_missing = int(df.isna().sum().sum())

    columns: List[Dict] = []
    assoc_cols: List[str] = []
    alerts: List[Dict] = []

    for name in df.columns:
        series = df[name]
        non_null = series.dropna()
        missing = int(series.isna().sum())
        n_unique = int(non_null.nunique())
        dtype = _column_dtype(series, n_unique)

        col: Dict = {
            "name": str(name),
            "dtype": dtype,
            "missing": missing,
            "missing_pct": round(100 * missing / n_rows, 2) if n_rows else 0.0,
            "unique": n_unique,
            "top_values": _top_values(non_null, n_rows) if n_rows else [],
            "stats": None,
        }
        if dtype == "numeric" and len(non_null):
            col["stats"] = {
                "min": float(non_null.min()), "max": float(non_null.max()),
                "mean": float(non_null.mean()), "std": float(non_null.std(ddof=0)),
                "median": float(non_null.median()),
            }
            col["histogram"] = _histogram(non_null)

        # ¿Apta para la matriz de asociación / como variable categórica?
        categorical_like = 2 <= n_unique <= MAX_CATEGORIES_FOR_ASSOC
        col["categorical_like"] = categorical_like

        # Índice de balance (evenness) para variables categóricas.
        if categorical_like:
            vc = non_null.value_counts(dropna=True).to_numpy()
            evenness = _normalized_entropy(vc)
            col["balance"] = {"evenness": round(evenness, 3), "label": _balance_label(evenness)}
        columns.append(col)
        if categorical_like:
            assoc_cols.append(str(name))

        # --- Alertas por columna ---
        if col["missing_pct"] >= 20:
            alerts.append({"level": "critical", "type": "missing", "columns": [str(name)],
                           "message": f"'{name}' tiene {col['missing_pct']:.0f}% de valores faltantes."})
        elif col["missing_pct"] >= 5:
            alerts.append({"level": "warning", "type": "missing", "columns": [str(name)],
                           "message": f"'{name}' tiene {col['missing_pct']:.0f}% de valores faltantes."})

        if n_unique <= 1:
            alerts.append({"level": "warning", "type": "constant", "columns": [str(name)],
                           "message": f"'{name}' es constante (un solo valor); no aporta al análisis."})
        if categorical_like and col["top_values"]:
            top = col["top_values"][0]
            if top["pct"] >= 90:
                alerts.append({"level": "warning", "type": "imbalance", "columns": [str(name)],
                               "message": f"'{name}' está muy desbalanceada: '{top['value']}' concentra "
                                          f"el {top['pct']:.0f}% de los casos."})

    # --- Roles sugeridos (necesitan ver todas las columnas) ---
    _assign_roles(df, columns, n_rows)
    roles = {c["name"]: c["role_hint"] for c in columns}
    actual_col = next((n for n, r in roles.items() if r == "actual"), None)
    pos_value = _positive_value(df[actual_col]) if actual_col else None

    for col in columns:
        name = col["name"]
        if roles[name] == "id":
            if col["unique"] == n_rows:
                msg = (f"'{name}' tiene un valor distinto en cada fila ({_fmt_int(n_rows)}). "
                       f"Parece un identificador: no la uses como variable protegida.")
            else:
                msg = (f"'{name}' tiene {_fmt_int(col['unique'])} valores distintos en "
                       f"{_fmt_int(n_rows)} filas. Parece un identificador: no la uses como "
                       f"variable protegida.")
            alerts.append({"level": "info", "type": "high_cardinality", "columns": [name],
                           "message": msg})

        # Grupos con pocos casos: solo en columnas que podrían ser variables protegidas.
        if roles[name] not in ("protected", "unused") or not col["categorical_like"]:
            continue
        sub = df[[name]].copy()
        if pos_value is not None:
            sub["_pos"] = pd.to_numeric(df[actual_col], errors="coerce") == pos_value
        small = []
        for tv in col["top_values"]:
            if tv.get("is_aggregate"):
                continue
            n = tv["count"]
            npos = None
            if pos_value is not None:
                mask = sub[name].astype(str) == str(tv["value"])
                npos = int(sub.loc[mask, "_pos"].sum())
            if n < min_group_size or (npos is not None and min(npos, n - npos) < min_group_size):
                txt = f"{tv['value']} ({_fmt_int(n)} casos"
                txt += f", {_fmt_int(npos)} positivos)" if npos is not None else ")"
                small.append(txt)
        if small:
            alerts.append({"level": "warning", "type": "small_group", "columns": [name],
                           "message": f"{name} tiene grupos con pocos casos: {_join_es(small[:5])}. "
                                      f"Sus resultados serán poco confiables: las tasas que se calculen "
                                      f"sobre menos de {min_group_size} casos no se usarán en el veredicto."})

    # --- Matriz de asociación (Cramér's V) entre columnas categórico-like ---
    matrix: List[List[float]] = []
    if len(assoc_cols) >= 2:
        cat_df = df[assoc_cols].astype("object")
        for a in assoc_cols:
            row = []
            for b in assoc_cols:
                if a == b:
                    row.append(1.0)
                else:
                    mask = cat_df[a].notna() & cat_df[b].notna()
                    v = cramers_v(cat_df.loc[mask, a], cat_df.loc[mask, b]) if mask.any() else 0.0
                    row.append(round(v, 3))
            matrix.append(row)

        # Alertas de posible proxy (asociación fuerte entre dos variables).
        # Solo entre una variable protegida y otra columna: que la predicción se
        # parezca al valor real es lo esperado (el modelo acierta).
        for i in range(len(assoc_cols)):
            for j in range(i + 1, len(assoc_cols)):
                v = matrix[i][j]
                a, b = assoc_cols[i], assoc_cols[j]
                ra, rb = roles.get(a), roles.get(b)
                if v < 0.5 or "protected" not in (ra, rb):
                    continue
                if {ra, rb} & {"prediction", "actual", "outcome", "id"}:
                    continue
                level = "warning" if v >= 0.7 else "info"
                if ra == "protected" and rb == "protected":
                    detalle = "Una podría reemplazar a la otra de forma indirecta (proxy) y esconder discriminación."
                else:
                    prot, otra = (a, b) if ra == "protected" else (b, a)
                    detalle = (f"'{otra}' podría reemplazar a '{prot}' de forma indirecta (proxy) "
                               f"y esconder discriminación.")
                alerts.append({"level": level, "type": "proxy", "columns": [a, b],
                               "message": f"'{a}' y '{b}' están muy relacionadas (V de Cramér = "
                                          f"{_fmt_dec(v)}). {detalle}"})

    # --- Tablas de contingencia precomputadas (bivariado / sesgo / intersección) ---
    # Solo columnas categórico-like con pocas categorías, para acotar el tamaño.
    crosstab_cols = [c["name"] for c in columns
                     if c["categorical_like"] and c["unique"] <= MAX_CATEGORIES_FOR_CROSSTAB]
    crosstabs: Dict[str, Dict] = {}
    cats_map = {c: _capped_categories(df[c].dropna()) for c in crosstab_cols}
    for a in crosstab_cols:
        for b in crosstab_cols:
            if a == b:
                continue
            xa, yb = cats_map[a], cats_map[b]
            sub = df[[a, b]].dropna().astype("object")
            sub = sub[sub[a].astype(str).isin(xa) & sub[b].astype(str).isin(yb)]
            ct = pd.crosstab(sub[a].astype(str), sub[b].astype(str))
            ct = ct.reindex(index=xa, columns=yb, fill_value=0)
            crosstabs[f"{a}|||{b}"] = {
                "x_values": xa,
                "y_values": yb,
                "counts": ct.to_numpy().astype(int).tolist(),
            }

    # Orden de alertas: critical > warning > info.
    order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: order.get(a["level"], 3))

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_cells_pct": round(100 * total_missing / total_cells, 2),
        "columns": columns,
        "associations": {"columns": assoc_cols, "matrix": matrix},
        "crosstabs": crosstabs,
        "crosstab_columns": crosstab_cols,
        "alerts": alerts,
        "min_group_size": min_group_size,
    }
