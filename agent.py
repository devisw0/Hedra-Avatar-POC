from dotenv import load_dotenv
import os, json, asyncio, math, re
import requests
from datetime import datetime, timedelta
from collections import Counter
import csv

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, RunContext, function_tool
from livekit.plugins.openai.realtime import RealtimeModel
from livekit.plugins import hedra

# ---- NEW: analysis deps
import pandas as pd
# Make fuzzy matching optional so the worker doesn't crash if rapidfuzz isn't installed
try:
    from rapidfuzz import process as fuzzprocess
except Exception:
    class _NoFuzz:
        @staticmethod
        def extractOne(*args, **kwargs):
            return None  # signal: no fuzzy available
    fuzzprocess = _NoFuzz()

load_dotenv()

# =============================================================================
#                                WEATHER HELPERS
# =============================================================================
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")

def _geocode_city(q: str) -> dict | None:
    """
    Use OpenWeather Direct Geocoding to resolve a free-text place into lat/lon.
    Docs: https://openweathermap.org/api/geocoding-api
    Returns: {"lat": float, "lon": float, "name": "Hillsborough, NJ, US"} or None
    """
    if not OPENWEATHER_KEY:
        raise RuntimeError("Missing OPENWEATHER_API_KEY in .env")
    q = q.strip()
    if not q:
        return None

    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": q, "limit": 1, "appid": OPENWEATHER_KEY}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    arr = r.json() or []
    if not arr:
        return None

    item = arr[0]
    parts = [p for p in [item.get("name"), item.get("state"), item.get("country")] if p]
    return {
        "lat": item.get("lat"),
        "lon": item.get("lon"),
        "name": ", ".join(parts) if parts else q,
    }

def _fetch_current_weather_latlon(lat: float, lon: float, units: str = "imperial") -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY, "units": units}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def _fetch_forecast_latlon(lat: float, lon: float, units: str = "imperial") -> dict:
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY, "units": units}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# =============================================================================
#                              FRAGRANCE ANALYSIS
# =============================================================================

# You can override this via env var if the file lives elsewhere
FRAG_CSV_PATH = os.getenv("FRAGRANCE_CSV_PATH", "fra_cleaned.csv")

# Columns we will look for (case-insensitive). We'll map whatever we find.
_FRAG_COLUMN_ALIASES = {
    "name": {"name", "perfume_name", "title", "fragrance", "scent", "perfume"},
    "brand": {"brand", "house", "brand_name"},
    "country": {"country", "origin", "brand_country", "country_of_origin", "made_in"},
    "gender": {"gender", "target", "for", "sex", "target_audience"},
    "family": {"family", "olfactory_family", "fragrance_family"},
    "season": {"season", "best_season"},
    "year": {"year", "launch_year", "release_year"},
    "rating": {"rating", "score", "rating_avg", "average_rating", "rating value", "rating_value"},
    "rating_count": {"rating_count", "votes", "num_votes", "reviews", "review_count", "number_of_votes", "rating count", "ratingcount"},
    "price": {"price", "price_usd", "usd", "cost"},
    "longevity": {"longevity"},
    "sillage": {"sillage"},
    "notes": {"notes", "note", "pyramid_notes", "note_pyramid"},   # (we'll also synthesize from Top/Middle/Base)
    "accords": {"accords", "main_accords", "accord"},               # (we'll also synthesize from mainaccord1..5)
}

def _find_col(df: pd.DataFrame, candidates: set[str]) -> str | None:
    """
    Map a logical name to a concrete df column.
    Strategy:
      1) exact case-insensitive match
      2) normalized match (strip spaces/_/-, lowercased)
      3) optional fuzzy per-candidate if rapidfuzz is available
    """
    if df is None or df.empty or len(df.columns) == 0:
        return None

    raw_cols = list(map(str, df.columns))
    raw_cols_lower = {c.lower(): c for c in raw_cols}

    def canon(s: str) -> str:
        # remove spaces/underscores/dashes/periods, lowercase
        return re.sub(r"[ \t_\-\.]+", "", s.strip().lower())

    norm_map = {canon(c): c for c in raw_cols}

    # 1) exact case-insensitive
    for want in candidates:
        w = want.strip()
        if not w:
            continue
        if w.lower() in raw_cols_lower:
            return raw_cols_lower[w.lower()]

    # 2) normalized exact
    for want in candidates:
        cw = canon(want)
        if cw in norm_map:
            return norm_map[cw]

    # 3) fuzzy per-candidate
    if hasattr(fuzzprocess, "extractOne") and fuzzprocess.extractOne is not None:
        choices = list(norm_map.keys())
        best_score = -1
        best_key = None
        for want in candidates:
            cw = canon(want)
            match = fuzzprocess.extractOne(cw, choices)
            if match and isinstance(match, (tuple, list)) and len(match) >= 2:
                cand_key, score = match[0], match[1]
                if score > best_score:
                    best_score = score
                    best_key = cand_key
        if best_key is not None and best_score >= 85:  # threshold
            return norm_map[best_key]

    return None

def _normalize_text(val) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None

def _canon_text(val) -> str | None:
    s = _normalize_text(val)
    return s.lower() if s is not None else None

_SPLIT_RE = re.compile(r'[,/]| and ', flags=re.IGNORECASE)

def _split_to_list(cell) -> list[str]:
    if pd.isna(cell) or cell is None:
        return []
    parts = _SPLIT_RE.split(str(cell))
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p.lower())
    return out

def _canon_gender(s: str | None) -> str | None:
    if not s:
        return None
    s = s.lower().strip()
    if s in {"men", "male", "man", "mens", "men's"}:
        return "male"
    if s in {"women", "female", "woman", "womens", "women's", "ladies"}:
        return "female"
    if s in {"unisex", "uni", "both"}:
        return "unisex"
    return s  # leave as-is if unfamiliar; still filterable

def _coerce_float(x):
    """
    Robust float coercion:
    - Handles European decimals (e.g., '1,42' -> 1.42)
    - Strips currency/units (e.g., '$1,234.50', '€1.234,50', '12.5 ml')
    - Handles thousands separators (either comma or dot)
    Returns float or None.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None

    # Keep digits, comma, dot, minus
    s = re.sub(r"[^\d,.\-]", "", s)

    # If we see both comma and dot, assume comma is thousands sep:
    #  "1,234.56" -> "1234.56"
    #  "1.234,56" -> treat dot as thousands, comma as decimal
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")

    try:
        return float(s)
    except Exception:
        return None

def _coerce_int(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return int(float(x))
    except Exception:
        return None

def _read_csv_resilient(path: str) -> pd.DataFrame:
    """
    Very forgiving CSV loader:
    - Tries multiple encodings WITH the python engine + sep inference
    - If that fails, sniffs delimiter/quotechar from a sample
    - Final fallback: open() with errors='replace' and let python engine parse
    """
    def _try_read(**kwargs):
        return pd.read_csv(path, dtype=str, on_bad_lines="skip", **kwargs)

    # Pass 1: python engine + sep=None (delimiter inference) across common encodings
    for enc in (None, "utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            kw = {"engine": "python", "sep": None}
            if enc:
                kw["encoding"] = enc
            return _try_read(**kw)
        except Exception as e:
            print(f"[FRAG] python-engine read (enc={enc or 'default'}) failed: {e}")

    # Pass 2: sniff delimiter & quotechar from a small sample, then read with python engine
    try:
        with open(path, "rb") as fb:
            head = fb.read(64 * 1024)
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                sample = head.decode(enc, errors="replace")
                dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
                sep = dialect.delimiter
                quotechar = dialect.quotechar
                print(f"[FRAG] sniffed sep='{sep}' quotechar='{quotechar}' using enc={enc}")
                return pd.read_csv(
                    path,
                    dtype=str,
                    engine="python",
                    sep=sep,
                    quotechar=quotechar,
                    encoding=enc,
                    on_bad_lines="skip",
                )
            except Exception as e:
                print(f"[FRAG] sniffer (enc={enc}) failed: {e}")
    except Exception as e:
        print(f"[FRAG] sniffer pre-read failed: {e}")

    # Pass 3 (final): open with errors='replace' and let python engine infer sep
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            print(f"[FRAG] final fallback: open(..., encoding={enc}, errors='replace')")
            with open(path, "r", encoding=enc, errors="replace") as f:
                return pd.read_csv(f, dtype=str, engine="python", sep=None, on_bad_lines="skip")
        except Exception as e:
            print(f"[FRAG] final fallback (enc={enc}) failed: {e}")

    print("[FRAG] all parsing strategies failed; returning empty DataFrame")
    return pd.DataFrame()

def _ensure_loaded_dataset() -> tuple[pd.DataFrame, dict]:
    """
    Load and normalize the fragrance dataset once.
    Returns (cleaned DataFrame, column map {logical_name: df_column_name}).
    """
    abs_path = os.path.abspath(FRAG_CSV_PATH)
    if not os.path.exists(abs_path):
        print(f"[FRAG] CSV not found at {abs_path}")
        return pd.DataFrame(), {}

    # ---- Load
    df = _read_csv_resilient(abs_path)
    if df.empty:
        print("[FRAG] dataframe is empty after read")
        return pd.DataFrame(), {}
    
    df.columns = [str(c).strip() for c in df.columns]  # trim header whitespace
    print(f"[FRAG] loaded file: {abs_path} shape={df.shape}")
    print(f"[FRAG] CSV columns detected: {list(df.columns)[:25]}{' ...' if len(df.columns) > 25 else ''}")

    # ---- Map columns using aliases (case/space/underscore tolerant via _find_col)
    colmap = {}
    for logical, candidates in _FRAG_COLUMN_ALIASES.items():
        found = _find_col(df, candidates)
        if found:
            colmap[logical] = found
    print(f"[FRAG] column map (pre-synthesis): {colmap}")

    # ---- Synthesize notes from Top/Middle/Base (if present)
    lower_cols = {c.lower(): c for c in df.columns}
    top_col  = next((lower_cols[k] for k in lower_cols if k in {"top", "top notes", "topnotes"}), None)
    mid_col  = next((lower_cols[k] for k in lower_cols if k in {"middle", "middle notes", "heart", "heart notes"}), None)
    base_col = next((lower_cols[k] for k in lower_cols if k in {"base", "base notes", "basenotes"}), None)

    def _split_or_empty(col_name):
        if col_name and col_name in df.columns:
            return df[col_name].map(_split_to_list)
        return pd.Series([[]] * len(df))

    if top_col or mid_col or base_col:
        parts = [_split_or_empty(top_col), _split_or_empty(mid_col), _split_or_empty(base_col)]
        df["__notes_list"] = [list({*a, *b, *c}) for a, b, c in zip(parts[0], parts[1], parts[2])]
        colmap.setdefault("notes", "__notes_list")

    # ---- Synthesize accords from mainaccord1..5
    acc_cols = [lower_cols[c] for c in lower_cols if c.startswith("mainaccord")]
    if acc_cols:
        def _row_acc(row):
            vals = []
            for c in acc_cols:
                v = row.get(c)
                if pd.notna(v) and str(v).strip():
                    v2 = str(v).strip().lower()
                    if v2 not in vals:
                        vals.append(v2)
            return vals
        df["__accords_list"] = df.apply(_row_acc, axis=1)
        colmap.setdefault("accords", "__accords_list")

    print(f"[FRAG] column map (final): {colmap}")

    # ---- Display-friendly originals (SET BEFORE normalization so casing is preserved)
    if "brand" in colmap and colmap["brand"] in df.columns:
        df["__brand_disp"] = df[colmap["brand"]].copy()
    else:
        df["__brand_disp"] = None
        
    if "name" in colmap and colmap["name"] in df.columns:
        df["__name_disp"] = df[colmap["name"]].copy()
    else:
        df["__name_disp"] = None

    # ---- Normalize text fields (lowercase for matching/filtering)
    for key in ("name", "brand", "country", "family", "season"):
        col = colmap.get(key)
        if col and col in df.columns:
            df[col] = df[col].map(_canon_text)

    # ---- Gender normalization
    gcol = colmap.get("gender")
    if gcol and gcol in df.columns:
        df[gcol] = df[gcol].map(_canon_gender)

    # ---- Year / decade
    ycol = colmap.get("year")
    if ycol and ycol in df.columns:
        df[ycol] = df[ycol].map(_coerce_int)
        df["__decade"] = df[ycol].map(lambda y: (y // 10) * 10 if isinstance(y, int) else None)

    # ---- Numeric columns
    for key in ("rating", "rating_count", "price", "longevity", "sillage"):
        col = colmap.get(key)
        if col and col in df.columns:
            df[col] = df[col].map(_coerce_float)

    # ---- If raw "notes"/"accords" existed, split them if synthesized lists aren't present
    if "notes" in colmap and colmap["notes"] in df.columns and "__notes_list" not in df.columns:
        df["__notes_list"] = df[colmap["notes"]].map(_split_to_list)
    if "accords" in colmap and colmap["accords"] in df.columns and "__accords_list" not in df.columns:
        df["__accords_list"] = df[colmap["accords"]].map(_split_to_list)

    print(f"[FRAG] normalization complete (rows={len(df)})")
    return df, colmap

# =============================================================================
#                                     AGENT
# =============================================================================

class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a concise, helpful voice assistant.\n"
                "Weather:\n"
                "- For current weather questions, call 'get_weather'.\n"
                "- For future weather (tomorrow/next few days), call 'get_forecast'.\n"
                "- If a weather tool returns error 'missing_city', ask which city.\n"
                "- If the user omits a city but a previous city is known, reuse that city.\n"
                "- Default units to Fahrenheit (imperial) if not specified.\n"
                "\n"
                "Fragrances (Fragrantica dataset):\n"
                "- For averages, use 'frag_get_average'.\n"
                "- For top/lowest by a numeric column, use 'frag_get_top_n' or 'frag_get_bottom_n'.\n"
                "- For 'which X has the most', use 'frag_count_items'.\n"
                "- For note/accord queries, use 'frag_find_by_notes'.\n"
                "- For price extremes, use 'frag_get_cheapest' or 'frag_get_most_expensive'.\n"
                "- When filters mention men/men's, treat as gender=male; women/ladies → female; unisex is supported.\n"
                "- If a brand or country is provided, filter by it (case-insensitive). "
                "If multiple constraints are given (e.g., notes + brand + price), pass them all via the tool filters.\n"
                "- If a tool returns no results, briefly explain why (e.g., no matches) and suggest relaxing filters."
            )
        )

        # Weather session memory
        self.last_place = None  # {"name": str, "lat": float, "lon": float}
        self.last_units = "imperial"

        # Fragrance data (loaded once)
        self._frag_df, self._frag_cols = _ensure_loaded_dataset()

    # ---------------------------
    # WEATHER TOOLS
    # ---------------------------
    @function_tool(description="Get current weather for a place (e.g., 'Hillsborough, NJ'). Units: 'imperial' or 'metric'.")
    async def get_weather(self, context: RunContext, city: str = "", units: str = "imperial") -> dict:
        units = (units or self.last_units or "imperial").lower()
        if units not in ("imperial", "metric"):
            units = "imperial"
        self.last_units = units

        place = None
        if city.strip():
            place = _geocode_city(city)
            if not place:
                return {"error": "missing_city", "message": "I couldn't find that place. Could you rephrase or add state/country?"}
            self.last_place = place
        elif self.last_place:
            place = self.last_place
        else:
            return {"error": "missing_city", "message": "Please tell me which city you'd like the weather for."}

        try:
            data = _fetch_current_weather_latlon(place["lat"], place["lon"], units)
            wind_speed = (data.get("wind") or {}).get("speed")
            wind_unit = "mph" if units == "imperial" else "m/s"
            return {
                "resolved_place": place["name"],
                "conditions": (data.get("weather") or [{}])[0].get("description", "unknown"),
                "temp": (data.get("main") or {}).get("temp"),
                "feels_like": (data.get("main") or {}).get("feels_like"),
                "humidity": (data.get("main") or {}).get("humidity"),
                "wind_speed": wind_speed,
                "wind_unit": wind_unit,
                "units": units,
            }
        except requests.HTTPError as e:
            return {"error": f"OpenWeather error: {e.response.status_code} {e.response.text[:120]}"}
        except Exception as e:
            return {"error": f"Weather lookup failed: {str(e)}"}

    @function_tool(description="Get a simple forecast for 'tomorrow' for a place. Units: 'imperial' or 'metric'.")
    async def get_forecast(self, context: RunContext, city: str = "", units: str = "imperial", when: str = "tomorrow") -> dict:
        units = (units or self.last_units or "imperial").lower()
        if units not in ("imperial", "metric"):
            units = "imperial"
        self.last_units = units

        place = None
        if city.strip():
            place = _geocode_city(city)
            if not place:
                return {"error": "missing_city", "message": "I couldn't find that place. Could you rephrase or add state/country?"}
            self.last_place = place
        elif self.last_place:
            place = self.last_place
        else:
            return {"error": "missing_city", "message": "Please tell me which city you'd like the forecast for."}

        try:
            fc = _fetch_forecast_latlon(place["lat"], place["lon"], units)
            today_utc = datetime.utcnow().date()
            target_date = today_utc + timedelta(days=1)

            entries = []
            for item in fc.get("list", []):
                ts = item.get("dt_txt")
                if not ts:
                    continue
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                if dt.date() == target_date:
                    entries.append(item)

            if not entries:
                return {"error": "no_forecast", "message": "I couldn't find tomorrow's forecast windows for that location."}

            temps = [(e.get("main") or {}).get("temp") for e in entries if (e.get("main") or {}).get("temp") is not None]
            temps = [t for t in temps if isinstance(t, (int, float))]
            descs = [((e.get("weather") or [{}])[0].get("description") or "").lower() for e in entries]
            tmin = min(temps) if temps else None
            tmax = max(temps) if temps else None
            common_desc = (Counter([d for d in descs if d]).most_common(1)[0][0]) if any(descs) else "unknown"

            return {
                "resolved_place": place["name"],
                "for_date": str(target_date),
                "summary": common_desc,
                "temp_min": tmin,
                "temp_max": tmax,
                "units": units,
            }

        except requests.HTTPError as e:
            return {"error": f"OpenWeather error: {e.response.status_code} {e.response.text[:120]}"}
        except Exception as e:
            return {"error": f"Forecast lookup failed: {str(e)}"}

    # ---------------------------
    # FRAGRANCE FILTERING CORE
    # ---------------------------
    def _frag_ready(self) -> bool:
        return not self._frag_df.empty and bool(self._frag_cols)

    def _apply_filters(self, filters: dict, *, match_all_notes: bool=False, match_all_accords: bool=False) -> pd.DataFrame:
        """
        Centralized filter helper. Supports keys:
          brand, country, gender, family, season, decade, year_min, year_max,
          price_min, price_max, rating_min, rating_max, votes_min,
          longevity_min, sillage_min,
          notes (list[str]), accords (list[str])
        """
        df = self._frag_df
        if df.empty:
            return df

        cm = self._frag_cols
        out = df.copy()  # Make a copy to avoid modifying original

        def _contains_all(cell_list: list[str], wants: list[str]) -> bool:
            s = set(cell_list or [])
            return all(w in s for w in wants)

        def _contains_any(cell_list: list[str], wants: list[str]) -> bool:
            s = set(cell_list or [])
            return any(w in s for w in wants)

        # brand / country / family / season
        for key, canon in (("brand","brand"),("country","country"),("family","family"),("season","season")):
            val = filters.get(key)
            col = cm.get(canon)
            if val and col and col in out.columns:
                v = str(val).strip().lower()
                out = out[out[col] == v]

        # gender (normalized)
        gval = filters.get("gender")
        gcol = cm.get("gender")
        if gval and gcol and gcol in out.columns:
            out = out[out[gcol] == _canon_gender(str(gval))]

        # decade (derived)
        dec = filters.get("decade")
        if dec is not None and "__decade" in out.columns:
            try:
                d = int(dec)
                out = out[out["__decade"] == d]
            except Exception:
                pass

        # year range
        ycol = cm.get("year")
        if ycol and ycol in out.columns:
            if filters.get("year_min") is not None:
                out = out[pd.to_numeric(out[ycol], errors='coerce') >= float(filters["year_min"])]
            if filters.get("year_max") is not None:
                out = out[pd.to_numeric(out[ycol], errors='coerce') <= float(filters["year_max"])]

        # numeric ranges
        for key, colkey in (("price_min","price"),("price_max","price"),
                            ("rating_min","rating"),("rating_max","rating"),
                            ("votes_min","rating_count"),
                            ("longevity_min","longevity"),("sillage_min","sillage")):
            val = filters.get(key)
            colname = cm.get(colkey)
            if val is not None and colname and colname in out.columns:
                numeric_col = pd.to_numeric(out[colname], errors='coerce')
                if key.endswith("_min"):
                    out = out[numeric_col >= float(val)]
                elif key.endswith("_max"):
                    out = out[numeric_col <= float(val)]

        # notes / accords (lists)
        notes = filters.get("notes") or []
        if notes and "__notes_list" in out.columns:
            wants = [str(n).strip().lower() for n in notes if str(n).strip()]
            if wants:
                if match_all_notes:
                    out = out[out["__notes_list"].map(lambda lst: _contains_all(lst, wants))]
                else:
                    out = out[out["__notes_list"].map(lambda lst: _contains_any(lst, wants))]

        accords = filters.get("accords") or []
        if accords and "__accords_list" in out.columns:
            wants = [str(a).strip().lower() for a in accords if str(a).strip()]
            if wants:
                if match_all_accords:
                    out = out[out["__accords_list"].map(lambda lst: _contains_all(lst, wants))]
                else:
                    out = out[out["__accords_list"].map(lambda lst: _contains_any(lst, wants))]

        return out

    def _rows_to_brief(self, rows: pd.DataFrame, limit: int | None = None) -> list[dict]:
        if rows is None or rows.empty:
            return []
        if limit:
            rows = rows.head(limit)
        cm = self._frag_cols

        # Prefer original-casing display columns if present
        name_col  = "__name_disp"  if "__name_disp"  in rows.columns else cm.get("name")
        brand_col = "__brand_disp" if "__brand_disp" in rows.columns else cm.get("brand")
        rating_col = cm.get("rating")
        rating_count_col = cm.get("rating_count")
        price_col = cm.get("price")

        def _to_float(v):
            try:
                return float(v)
            except Exception:
                return None

        def _to_int(v):
            try:
                return int(float(v))
            except Exception:
                return None

        out = []
        for _, r in rows.iterrows():
            item = {
                "name":  (r.get(name_col) if name_col and name_col in r.index else None),
                "brand": (r.get(brand_col) if brand_col and brand_col in r.index else None),
            }
            if rating_col and rating_col in r.index:
                item["rating"] = _to_float(r.get(rating_col))
            if rating_count_col and rating_count_col in r.index:
                item["rating_count"] = _to_int(r.get(rating_count_col))
            if price_col and price_col in r.index:
                item["price"] = _to_float(r.get(price_col))
            out.append(item)
        return out

    # ---------------------------
    # FRAGRANCE TOOLS
    # ---------------------------
    @function_tool(description="Fragrance: compute an average for a numeric column (e.g., rating, price, longevity). Filters supported: brand, country, gender, family, season, decade, notes, accords, price_min/max, rating_min/max, votes_min, longevity_min, sillage_min.")
    async def frag_get_average(self, context: RunContext, column: str, filters: dict | None = None, match_all_notes: bool = False, match_all_accords: bool = False) -> dict:
        """
        Returns: {"average": float | None, "count": int}
        """
        if not self._frag_ready():
            return {"error": f"Fragrance dataset not found at {FRAG_CSV_PATH}."}
        filters = filters or {}
        cm = self._frag_cols
        col = cm.get(column.lower()) or cm.get(column) or column
        df2 = self._apply_filters(filters, match_all_notes=match_all_notes, match_all_accords=match_all_accords)
        if df2.empty or col not in df2.columns:
            return {"average": None, "count": 0}
        series = pd.to_numeric(df2[col], errors="coerce").dropna()
        if series.empty:
            return {"average": None, "count": 0}
        return {"average": float(series.mean()), "count": int(series.count())}

    @function_tool(description="Fragrance: return top-N rows by a numeric column (e.g., rating, rating_count, price). Include filters like brand, country, notes, gender, etc.")
    async def frag_get_top_n(self, context: RunContext, column: str, filters: dict | None = None, n: int = 5, ascending: bool = False, match_all_notes: bool = False, match_all_accords: bool = False) -> dict:
        if not self._frag_ready():
            return {"error": f"Fragrance dataset not found at {FRAG_CSV_PATH}."}
        filters = filters or {}
        cm = self._frag_cols
        col = cm.get(column.lower()) or cm.get(column) or column
        df2 = self._apply_filters(filters, match_all_notes=match_all_notes, match_all_accords=match_all_accords)
        if df2.empty or col not in df2.columns:
            return {"items": []}
        # Re-coerce to numeric just-in-case before sorting (extra safety)
        df3 = df2.copy()
        df3[col] = pd.to_numeric(df3[col], errors="coerce")
        df3 = df3.dropna(subset=[col])  # Remove rows where the column is NaN
        df3 = df3.sort_values(by=col, ascending=ascending, na_position="last")
        return {"items": self._rows_to_brief(df3, limit=max(1, int(n)))}

    @function_tool(description="Fragrance: return bottom-N rows by a numeric column (e.g., price, rating). Include filters like brand, country, notes, gender, etc.")
    async def frag_get_bottom_n(self, context: RunContext, column: str, filters: dict | None = None, n: int = 5, match_all_notes: bool = False, match_all_accords: bool = False) -> dict:
        return await self.frag_get_top_n(context, column=column, filters=filters, n=n, ascending=True, match_all_notes=match_all_notes, match_all_accords=match_all_accords)

    @function_tool(description="Fragrance: count items grouped by a column (e.g., brand, country, gender, decade). Returns dict of counts after filters.")
    async def frag_count_items(self, context: RunContext, group_by: str, filters: dict | None = None, match_all_notes: bool = False, match_all_accords: bool = False) -> dict:
        if not self._frag_ready():
            return {"error": f"Fragrance dataset not found at {FRAG_CSV_PATH}."}
        filters = filters or {}
        df2 = self._apply_filters(filters, match_all_notes=match_all_notes, match_all_accords=match_all_accords)
        if df2.empty:
            return {"counts": {}}
        cm = self._frag_cols
        gb_col = "__decade" if group_by.lower() == "decade" else (cm.get(group_by.lower()) or cm.get(group_by) or group_by)
        if gb_col not in df2.columns:
            return {"counts": {}}
        vc = df2[gb_col].value_counts(dropna=True)
        return {"counts": {("" if pd.isna(k) else str(int(k)) if gb_col=="__decade" and pd.notna(k) else str(k)): int(v) for k, v in vc.items()}}

    @function_tool(description="Fragrance: find perfumes matching notes/accords. Set match_all=True to require all notes/accords. You can also add brand/country/gender/price filters.")
    async def frag_find_by_notes(self, context: RunContext, notes: list[str] | None = None, accords: list[str] | None = None, filters: dict | None = None, match_all: bool = False) -> dict:
        if not self._frag_ready():
            return {"error": f"Fragrance dataset not found at {FRAG_CSV_PATH}."}
        filters = filters or {}
        if notes:
            filters["notes"] = notes
        if accords:
            filters["accords"] = accords
        df2 = self._apply_filters(filters, match_all_notes=match_all, match_all_accords=match_all)
        return {"items": self._rows_to_brief(df2, limit=10)}

    @function_tool(description="Fragrance: get the cheapest item given filters (e.g., brand, gender, notes, rating_min).")
    async def frag_get_cheapest(self, context: RunContext, filters: dict | None = None, match_all_notes: bool = False, match_all_accords: bool = False) -> dict:
        if not self._frag_ready():
            return {"error": f"Fragrance dataset not found at {FRAG_CSV_PATH}."}
        filters = filters or {}
        cm = self._frag_cols
        price_col = cm.get("price") or "price"
        df2 = self._apply_filters(filters, match_all_notes=match_all_notes, match_all_accords=match_all_accords)
        if df2.empty or price_col not in df2.columns:
            return {"item": None}
        df3 = df2.copy()
        df3[price_col] = pd.to_numeric(df3[price_col], errors="coerce")
        df3 = df3.dropna(subset=[price_col])  # Remove rows where price is NaN
        if df3.empty:
            return {"item": None}
        df3 = df3.sort_values(by=price_col, ascending=True, na_position="last")
        items = self._rows_to_brief(df3, limit=1)
        return {"item": (items[0] if items else None)}

    @function_tool(description="Fragrance: get the most expensive item given filters (e.g., brand, gender, notes, rating_min).")
    async def frag_get_most_expensive(self, context: RunContext, filters: dict | None = None, match_all_notes: bool = False, match_all_accords: bool = False) -> dict:
        if not self._frag_ready():
            return {"error": f"Fragrance dataset not found at {FRAG_CSV_PATH}."}
        filters = filters or {}
        cm = self._frag_cols
        price_col = cm.get("price") or "price"
        df2 = self._apply_filters(filters, match_all_notes=match_all_notes, match_all_accords=match_all_accords)
        if df2.empty or price_col not in df2.columns:
            return {"item": None}
        df3 = df2.copy()
        df3[price_col] = pd.to_numeric(df3[price_col], errors="coerce")
        df3 = df3.dropna(subset=[price_col])  # Remove rows where price is NaN
        if df3.empty:
            return {"item": None}
        df3 = df3.sort_values(by=price_col, ascending=False, na_position="last")
        items = self._rows_to_brief(df3, limit=1)
        return {"item": (items[0] if items else None)}

# ---------------------------
# Worker entrypoint
# ---------------------------

async def entrypoint(ctx: agents.JobContext):
    # 1) CRITICAL: Connect to the room first
    await ctx.connect()  # REQUIRED so ctx.room is valid
    
    # 2) Create the realtime session (OpenAI Realtime handles STT/LLM/TTS)
    session = AgentSession(
        llm=RealtimeModel(),  # uses OPENAI_API_KEY
    )

    # 3) Start Hedra avatar (requires HEDRA_API_KEY and HEDRA_AVATAR_ID)
    avatar_id = os.getenv("HEDRA_AVATAR_ID")
    if not avatar_id:
        print("WARNING: HEDRA_AVATAR_ID not set, skipping avatar")
        avatar = None
    else:
        avatar = hedra.AvatarSession(avatar_id=avatar_id)
        print("Starting Hedra avatar…")
        await avatar.start(session, room=ctx.room)
        print("Hedra avatar started.")

    # 4) Start the agent
    await session.start(agent=Assistant(), room=ctx.room, room_input_options=RoomInputOptions())

    # 5) Optional greeting
    await session.generate_reply(instructions=(
        "Hi there! I can check the current weather or tomorrow's forecast, "
        "and I can also answer questions about the fragrance dataset—like averages, top items, and filters."
    ))

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))