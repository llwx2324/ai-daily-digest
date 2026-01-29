import os
import json
import re
import time
import csv
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import feedparser
from dateutil import parser as dtparser


CST = timezone(timedelta(hours=8))  # China Standard Time (UTC+8)


# -----------------------------
# HTTP utils: retry + degrade
# -----------------------------
def http_get_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
    retries: int = 2,
    backoff: float = 0.8,
) -> Optional[Any]:
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i < retries:
                time.sleep(backoff * (2 ** i))
            else:
                return None
    return None


# -----------------------------
# Config / helpers
# -----------------------------
def load_config() -> Dict[str, Any]:
    with open("config/sources.json", "r", encoding="utf-8") as f:
        return json.load(f)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def safe_parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for x in items:
        key = x.get("url") or ("t:" + norm(x.get("title", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


# -----------------------------
# Fetchers
# -----------------------------
def fetch_rss_items(
    rss_sources: List[Dict[str, str]],
    fetch_per_source: int = 60
) -> Tuple[List[Dict[str, Any]], List[str]]:
    items: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for src in rss_sources:
        name = src.get("name", "Unknown")
        url = src.get("url", "")
        try:
            feed = feedparser.parse(url)
            entries = getattr(feed, "entries", []) or []
            if not entries:
                warnings.append(f"RSS empty or unreadable: {name}")
                continue

            for e in entries[:fetch_per_source]:
                title = getattr(e, "title", "") or ""
                link = getattr(e, "link", "") or ""
                published = getattr(e, "published", None) or getattr(e, "updated", None)
                summary = getattr(e, "summary", "") or ""

                items.append({
                    "kind": "rss",
                    "source": name,
                    "title": title.strip(),
                    "url": link.strip(),
                    "summary": re.sub(r"\s+", " ", summary).strip(),
                    "published": safe_parse_dt(published),
                    "meta": {}
                })
        except Exception:
            warnings.append(f"RSS failed: {name}")
            continue

    return items, warnings


def fetch_hn_top(fetch_n: int = 50) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    top_ids = http_get_json(
        "https://hacker-news.firebaseio.com/v0/topstories.json",
        timeout=20,
        retries=2,
    )
    if not isinstance(top_ids, list):
        warnings.append("HN unavailable: failed to fetch topstories")
        return [], warnings

    out: List[Dict[str, Any]] = []
    for sid in top_ids[:fetch_n]:
        it = http_get_json(
            f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
            timeout=20,
            retries=1,
        )
        if not isinstance(it, dict) or it.get("type") != "story":
            continue

        title = it.get("title", "") or ""
        url = it.get("url") or f"https://news.ycombinator.com/item?id={sid}"
        score = int(it.get("score", 0) or 0)

        out.append({
            "kind": "hn",
            "source": "Hacker News",
            "title": title.strip(),
            "url": str(url).strip(),
            "summary": "",
            "published": None,
            "meta": {"hn_score": score}
        })

    return out, warnings


def github_search_repos(
    token: Optional[str],
    q: str,
    per_page: int = 30,
    sort: str = "stars",
    order: str = "desc"
) -> Optional[Dict[str, Any]]:
    url = "https://api.github.com/search/repositories"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    return http_get_json(
        url,
        headers=headers,
        params={"q": q, "sort": sort, "order": order, "per_page": per_page},
        timeout=25,
        retries=2,
    )


def fetch_github_trending_dual(
    token: Optional[str],
    since_date_utc: str,
    per_page: int = 30
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    P2: Dual-query trending proxy:
      - NEW: created:>since_date stars:>50
      - HOT: pushed:>since_date stars:>200 (captures older but actively updated repos)
    """
    warnings: List[str] = []

    q_new = f"created:>{since_date_utc} stars:>50"
    data_new = github_search_repos(token, q_new, per_page=per_page, sort="stars", order="desc")
    items_new = (data_new or {}).get("items", []) if isinstance(data_new, dict) else []
    if data_new is None:
        warnings.append("GitHub search failed: new-repos query")

    q_hot = f"pushed:>{since_date_utc} stars:>200"
    data_hot = github_search_repos(token, q_hot, per_page=per_page, sort="stars", order="desc")
    items_hot = (data_hot or {}).get("items", []) if isinstance(data_hot, dict) else []
    if data_hot is None:
        warnings.append("GitHub search failed: hot-repos query")

    out: List[Dict[str, Any]] = []
    for repo in items_new + items_hot:
        try:
            out.append({
                "name": repo["full_name"],
                "url": repo["html_url"],
                "stars": repo["stargazers_count"],
                "desc": (repo.get("description") or "").strip(),
            })
        except Exception:
            continue

    # dedupe by url
    seen = set()
    deduped = []
    for x in out:
        k = x.get("url")
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append(x)

    deduped.sort(key=lambda x: int(x.get("stars", 0) or 0), reverse=True)
    return deduped, warnings


# -----------------------------
# Scoring
# -----------------------------
def count_hits(text: str, words: List[str]) -> int:
    t = norm(text)
    hits = 0
    for w in words:
        w2 = norm(w)
        if w2 and w2 in t:
            hits += 1
    return hits


def score_item(item: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    scoring = cfg["scoring"]
    keywords = cfg.get("keywords", [])
    strong_keywords = cfg.get("strong_keywords", [])
    official_sources = set(cfg.get("official_sources", []))
    media_sources = set(cfg.get("media_sources", []))

    text = f"{item.get('title','')} {item.get('summary','')}"
    kw_hits = count_hits(text, keywords)
    strong_hits = count_hits(text, strong_keywords)

    s = 0
    s += kw_hits * scoring.get("keyword_hit", 2)
    s += strong_hits * scoring.get("strong_keyword_hit", 4)

    src = item.get("source", "")
    if src in official_sources:
        s += scoring.get("official_source_bonus", 6)
    elif src in media_sources:
        s += scoring.get("media_source_bonus", 2)

    if item.get("kind") == "hn":
        hn_score = int(item.get("meta", {}).get("hn_score", 0) or 0)
        if hn_score >= 200:
            s += scoring.get("hn_score_200_bonus", 6)
        elif hn_score >= 100:
            s += scoring.get("hn_score_100_bonus", 3)

    return s


# -----------------------------
# Trend module (v1): keyword counts + 7d delta
# -----------------------------
def build_trend_counts(
    items: List[Dict[str, Any]],
    keywords: List[str]
) -> Dict[str, int]:
    """
    Count keyword mentions in (title + summary). One item can hit multiple keywords.
    Matching is substring on normalized text (consistent with scoring).
    """
    keys = [norm(k) for k in keywords if norm(k)]
    counts = {k: 0 for k in keys}

    for it in items:
        text = norm(f"{it.get('title','')} {it.get('summary','')}")
        for k in keys:
            if k in text:
                counts[k] += 1
    return counts


def read_trend_csv(path: str) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """
    Returns (columns_without_date, data_by_date).
    """
    if not os.path.exists(path):
        return [], {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in (reader.fieldnames or []) if c and c != "date"]
        data: Dict[str, Dict[str, int]] = {}
        for row in reader:
            d = row.get("date")
            if not d:
                continue
            data[d] = {}
            for c in cols:
                try:
                    data[d][c] = int(row.get(c, "0") or 0)
                except Exception:
                    data[d][c] = 0
        return cols, data


def write_trend_csv(path: str, cols: List[str], data_by_date: Dict[str, Dict[str, int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # sort dates ascending
    dates = sorted(data_by_date.keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date"] + cols)
        writer.writeheader()
        for d in dates:
            row = {"date": d}
            for c in cols:
                row[c] = data_by_date.get(d, {}).get(c, 0)
            writer.writerow(row)


def compute_7d_delta(
    today: str,
    cols: List[str],
    data_by_date: Dict[str, Dict[str, int]],
    window_days: int = 7
) -> Dict[str, Dict[str, float]]:
    """
    For each keyword:
      delta = today_count - avg(previous <= window_days days)
    If insufficient history, avg is computed over whatever is available (excluding today).
    """
    # Build list of previous dates (exclude today), take last window_days by date
    prev_dates = [d for d in sorted(data_by_date.keys()) if d < today]
    prev_dates = prev_dates[-window_days:]

    out: Dict[str, Dict[str, float]] = {}
    for k in cols:
        today_count = float(data_by_date.get(today, {}).get(k, 0))
        if prev_dates:
            avg = sum(float(data_by_date.get(d, {}).get(k, 0)) for d in prev_dates) / float(len(prev_dates))
        else:
            avg = 0.0
        out[k] = {"today": today_count, "avg_prev": avg, "delta": (today_count - avg)}
    return out


def format_trend_watch(delta_map: Dict[str, Dict[str, float]], top_n: int = 8) -> List[str]:
    """
    Show top_n keywords by absolute delta; ties broken by today's count.
    """
    items = []
    for k, v in delta_map.items():
        items.append((k, v["delta"], v["today"], v["avg_prev"]))
    items.sort(key=lambda x: (abs(x[1]), x[2]), reverse=True)
    items = items[:top_n]

    lines = []
    for k, delta, today, avg_prev in items:
        if delta > 0:
            arrow = "▲"
        elif delta < 0:
            arrow = "▼"
        else:
            arrow = "—"
        # Keep it compact: show today and delta vs 7d avg
        lines.append(f"- {k}: {arrow} {delta:+.1f}（今日 {int(today)}，近7日均值 {avg_prev:.1f}）")
    return lines


# -----------------------------
# Rendering
# -----------------------------
def render_markdown(
    date_cst_str: str,
    top_signals: List[Dict[str, Any]],
    rss_items: List[Dict[str, Any]],
    gh_items: List[Dict[str, Any]],
    trend_lines: List[str],
    warnings: List[str],
) -> str:
    def md_link(title: str, url: str) -> str:
        return f"[{title}]({url})" if url else title

    lines: List[str] = []
    lines.append(f"# AI Daily Digest — {date_cst_str}\n")

    if warnings:
        lines.append("## ⚠️ 运行告警\n")
        for w in warnings[:12]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## 1) 今日必看（Top Signals）\n")
    if top_signals:
        for x in top_signals:
            extra = ""
            if x.get("kind") == "hn":
                extra = f" (HN {x.get('meta', {}).get('hn_score', 0)} pts)"
            lines.append(f"- {md_link(x.get('title',''), x.get('url',''))}{extra} — {x.get('source','')}")
    else:
        lines.append("-（无）")
    lines.append("")

    lines.append("## 2) 新闻源 RSS\n")
    if rss_items:
        for x in rss_items:
            lines.append(f"- {x.get('source','')}: {md_link(x.get('title',''), x.get('url',''))}")
    else:
        lines.append("-（无）")
    lines.append("")

    lines.append("## 3) GitHub Trending（Proxy）\n")
    if gh_items:
        for x in gh_items:
            desc = re.sub(r"\s+", " ", x.get("desc", ""))[:120]
            suffix = f" — {desc}" if desc else ""
            lines.append(f"- ⭐ {x.get('stars', 0)}: {md_link(x.get('name',''), x.get('url',''))}{suffix}")
    else:
        lines.append("-（无）")
    lines.append("")

    lines.append("## 4) Trend Watch（近 7 天）\n")
    if trend_lines:
        lines.extend(trend_lines)
    else:
        lines.append("-（暂无趋势数据）")
    lines.append("")

    lines.append("## 5) 一句话总结（可手动补）\n")
    lines.append("- 今天最值得关注的方向是：__________\n")

    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = load_config()
    limits = cfg["limits"]

    # robust time derivation (UTC -> CST)
    now_cst = datetime.now(timezone.utc).astimezone(CST)
    date_str = now_cst.strftime("%Y-%m-%d")

    since_dt_utc = datetime.now(timezone.utc) - timedelta(days=1)
    since_date_utc = since_dt_utc.strftime("%Y-%m-%d")

    token = os.getenv("GH_PAT") or os.getenv("GITHUB_TOKEN")

    warnings: List[str] = []

    # HN (degradable)
    hn_raw, hn_warn = fetch_hn_top(fetch_n=int(limits.get("hn_fetch", 80) or 80))
    warnings += hn_warn

    # RSS (per-source try/except)
    rss_raw, rss_warn = fetch_rss_items(
        cfg.get("rss", []),
        fetch_per_source=int(limits.get("rss_fetch_per_source", 60) or 60),
    )
    warnings += rss_warn

    # GitHub dual-query
    gh_raw, gh_warn = fetch_github_trending_dual(token, since_date_utc, per_page=30)
    warnings += gh_warn

    # RSS time filter only (drop stale)
    rss_recent: List[Dict[str, Any]] = []
    for x in rss_raw:
        pub = x.get("published")
        if pub and pub < since_dt_utc:
            continue
        rss_recent.append(x)

    # Score + sort for HN/RSS
    for x in hn_raw:
        x["score"] = score_item(x, cfg)
    for x in rss_recent:
        x["score"] = score_item(x, cfg)

    hn_sorted = sorted(
        hn_raw,
        key=lambda x: (x.get("score", 0), x.get("meta", {}).get("hn_score", 0)),
        reverse=True,
    )
    rss_sorted = sorted(
        rss_recent,
        key=lambda x: (x.get("score", 0), x.get("published") or datetime(1970, 1, 1, tzinfo=timezone.utc)),
        reverse=True,
    )

    # Truncate (display)
    hn_take = dedupe(hn_sorted)[:int(limits.get("hn_take", 12) or 12)]
    rss_take = dedupe(rss_sorted)[:int(limits.get("rss_take", 18) or 18)]
    gh_take = gh_raw[:int(limits.get("gh_take", 8) or 8)]

    # Top Signals from mixed pool (HN + RSS)
    mixed = dedupe(sorted(hn_take + rss_take, key=lambda x: x.get("score", 0), reverse=True))
    top_signals = mixed[:int(limits.get("top_signals", 5) or 5)]

    # -----------------------------
    # Trend update (v1)
    # -----------------------------
    # Trend counts should reflect the actual collected candidates (not only displayed truncation),
    # but using recent pool keeps it stable and cheap.
    trend_keywords = [norm(k) for k in cfg.get("keywords", []) if norm(k)]
    trend_pool = hn_raw + rss_recent
    today_counts = build_trend_counts(trend_pool, trend_keywords)

    trend_path = os.path.join("trend", "trend.csv")
    cols_existing, data_by_date = read_trend_csv(trend_path)

    # Keep columns stable: prefer existing order; append any new keywords to the end
    cols = cols_existing[:] if cols_existing else trend_keywords[:]
    for k in trend_keywords:
        if k not in cols:
            cols.append(k)

    # Write/replace today's row
    data_by_date[date_str] = {k: int(today_counts.get(k, 0)) for k in cols}

    write_trend_csv(trend_path, cols, data_by_date)

    # Prepare Trend Watch lines
    delta_map = compute_7d_delta(date_str, cols, data_by_date, window_days=7)
    trend_lines = format_trend_watch(delta_map, top_n=8)

    # -----------------------------
    # Render + output
    # -----------------------------
    md = render_markdown(date_str, top_signals, rss_take, gh_take, trend_lines, warnings)

    os.makedirs("daily", exist_ok=True)
    out_path = os.path.join("daily", f"{date_str}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote {out_path}")
    print(f"Updated {trend_path}")


if __name__ == "__main__":
    main()
