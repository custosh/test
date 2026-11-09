from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "Prime-pair hyperbola explorer"
EXACT_PAIR_LIMIT = 200_000
DEFAULT_SAMPLE_LIMIT = 100_000
SIEVE_MAX_BITS = 24
SMALL_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)


def sieve_primes_in_range(bit_count: int) -> list[int]:
    """Return every prime with exactly bit_count bits (practical up to 24 bits)."""
    lower = 1 << (bit_count - 1)
    upper = (1 << bit_count) - 1
    flags = np.ones(upper - lower + 1, dtype=np.bool_)
    for p in range(2, math.isqrt(upper) + 1):
        if p * p > upper:
            break
        # The first multiple in the interval, but never mark p itself.
        first = max(p * p, ((lower + p - 1) // p) * p)
        flags[first - lower :: p] = False
    return (np.flatnonzero(flags) + lower).astype(object).tolist()


def is_probable_prime(n: int, rounds: int, rng: random.Random) -> bool:
    """Miller-Rabin probable-prime test; adequate for visualization, not proof."""
    if n < 2:
        return False
    for p in (2,) + SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_probable_primes(bit_count: int, count: int, seed: int) -> list[int]:
    """Generate a reproducible pool of distinct probable primes of the requested size."""
    rng = random.Random(seed)
    lower = 1 << (bit_count - 1)
    upper = (1 << bit_count) - 1
    found: set[int] = set()
    # 16 Miller-Rabin rounds gives a very small false-positive probability.
    while len(found) < count:
        candidate = rng.getrandbits(bit_count) | lower | 1
        if candidate > upper:
            continue
        if candidate in found:
            continue
        if is_probable_prime(candidate, 16, rng):
            found.add(candidate)
    return sorted(found)


def row_start(i: int, m: int) -> int:
    return i * m - (i * (i - 1)) // 2


def unrank_pair(k: int, m: int) -> tuple[int, int]:
    discriminant = (2 * m + 1) ** 2 - 8 * k
    i = ((2 * m + 1) - math.isqrt(discriminant)) // 2
    while i > 0 and row_start(i, m) > k:
        i -= 1
    while i + 1 < m and row_start(i + 1, m) <= k:
        i += 1
    return i, i + (k - row_start(i, m))


def sampled_indices(total: int, count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if count >= total:
        return list(range(total))
    return rng.sample(range(total), count)


def make_pairs(primes: list[int], sample_limit: int, seed: int):
    m = len(primes)
    total = m * (m + 1) // 2
    exact = total <= EXACT_PAIR_LIMIT
    if exact:
        indices = range(total)
    else:
        indices = sampled_indices(total, min(sample_limit, total), seed)
    p_values, q_values = [], []
    for k in indices:
        i, j = unrank_pair(k, m)
        p_values.append(primes[i])
        q_values.append(primes[j])
    return total, exact, p_values, q_values


def safe_log2_int(value: int) -> float:
    return math.log2(value)


def build_plot(p_values, q_values, lower: int, bit_count: int, target_n: int):
    log2_n = safe_log2_int(target_n)
    x_norm = np.asarray([p / lower for p in p_values], dtype=float)
    y_norm = np.asarray([q / lower for q in q_values], dtype=float)
    log_ratio = np.empty(len(p_values), dtype=float)
    product_text = []
    for i, (p, q) in enumerate(zip(p_values, q_values)):
        product = p * q
        product_text.append(str(product))
        log_ratio[i] = 0.0 if product == target_n else safe_log2_int(product) - log2_n

    frame = pd.DataFrame({
        "x": x_norm, "y": y_norm, "log_ratio": log_ratio,
        "p": [str(v) for v in p_values], "q": [str(v) for v in q_values],
        "product": product_text,
    })
    fig = go.Figure()
    max_abs = float(np.max(np.abs(log_ratio))) if len(log_ratio) else 1.0
    if max_abs == 0:
        max_abs = 1.0
    fig.add_trace(go.Scattergl(
        x=frame["x"], y=frame["y"], mode="markers", name="Prime pairs",
        customdata=frame[["p", "q", "product", "log_ratio"]].to_numpy(),
        marker={"size": 6 if len(frame) < 50_000 else 3, "opacity": 0.75,
                "color": frame["log_ratio"], "colorscale": "RdBu_r", "cmid": 0,
                "cmin": -max_abs, "cmax": max_abs,
                "colorbar": {"title": "log₂(pq / N)"}},
        hovertemplate=("p = %{customdata[0]}<br>q = %{customdata[1]}<br>"
                       "p × q = %{customdata[2]}<br>"
                       "log₂(pq / N) = %{customdata[3]:.6g}<extra></extra>"),
    ))

    # Normalized ideal curve: y' = (N / lower²) / x'.
    x_curve = np.linspace(1.0, 2.0, 600)
    log2_scaled_n = log2_n - 2 * (bit_count - 1)
    curve_log_y = log2_scaled_n - np.log2(x_curve)
    curve_y = np.full_like(x_curve, np.nan)
    visible = (curve_log_y >= 0.0) & (curve_log_y <= 1.0)
    curve_y[visible] = np.exp2(curve_log_y[visible])
    fig.add_trace(go.Scatter(x=x_curve, y=curve_y, mode="lines",
        name="Ideal hyperbola: pq = N",
        line={"color": "#111827", "width": 3, "dash": "dash"},
        hovertemplate="Ideal curve: pq = N<extra></extra>"))
    fig.add_trace(go.Scatter(x=[1.0, 2.0], y=[1.0, 2.0], mode="lines",
        name="p = q", line={"color": "#64748b", "width": 1, "dash": "dot"},
        hoverinfo="skip"))
    fig.update_layout(
        template="plotly_white", height=720,
        title="Products of same-bit-length prime pairs",
        xaxis_title=f"p / 2^{bit_count - 1} (normalized p)",
        yaxis_title=f"q / 2^{bit_count - 1} (normalized q)",
        xaxis={"range": [0.98, 2.02], "constrain": "domain", "zeroline": False},
        yaxis={"range": [0.98, 2.02], "scaleanchor": "x", "scaleratio": 1, "zeroline": False},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 75, "r": 35, "t": 90, "b": 75},
    )
    return fig


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write("Every plotted point is an unordered pair of prime numbers p ≤ q "
             "with the selected bit length. Point colour represents the product "
             "relative to N; the dashed curve is the ideal relation p × q = N.")

    with st.sidebar:
        st.header("Parameters")
        bit_count = st.number_input("Bit count per prime", min_value=2, max_value=4096, value=8, step=1)
        suggested_n = 1 << (2 * (int(bit_count) - 1))
        st.caption(f"Prime range: 2^{int(bit_count)-1} to 2^{int(bit_count)} − 1. "
                   f"Suggested N: 2^{2 * (int(bit_count)-1)}")
        if "target_n_text" not in st.session_state:
            st.session_state.target_n_text = str(suggested_n)
        if "previous_suggested_n" not in st.session_state:
            st.session_state.previous_suggested_n = suggested_n
        elif st.session_state.previous_suggested_n != suggested_n:
            if st.session_state.target_n_text == str(st.session_state.previous_suggested_n):
                st.session_state.target_n_text = str(suggested_n)
            st.session_state.previous_suggested_n = suggested_n
        if st.button("Reset N to the minimum possible product", use_container_width=True):
            st.session_state.target_n_text = str(suggested_n)
        target_text = st.text_input("Target N (positive decimal integer)", key="target_n_text")
        if int(bit_count) <= SIEVE_MAX_BITS:
            st.caption("All primes in this bit range will be enumerated with a sieve.")
        else:
            prime_pool_size = st.slider("Number of probable primes to generate", 100, 5000, 500, 100)
            st.caption("For large bit counts, a reproducible pool is generated using Miller–Rabin.")
        sample_limit = st.slider("Maximum plotted pairs when sampling", 10_000, 250_000, DEFAULT_SAMPLE_LIMIT, 10_000)
        seed = st.number_input("Generation seed", min_value=0, value=42, step=1)
        generate = st.button("Generate graph", type="primary", use_container_width=True)

    if "has_generated" not in st.session_state:
        st.session_state.has_generated = False
    if generate:
        st.session_state.has_generated = True
        st.session_state.last_bit_count = int(bit_count)
        st.session_state.last_target_text = target_text
        st.session_state.last_sample_limit = int(sample_limit)
        st.session_state.last_seed = int(seed)
        st.session_state.last_prime_pool_size = int(prime_pool_size) if int(bit_count) > SIEVE_MAX_BITS else None

    if not st.session_state.has_generated:
        st.info("Choose the bit count and N in the sidebar, then select Generate graph.")
        return

    try:
        n_text = st.session_state.last_target_text.strip().replace("_", "")
        if not n_text or not n_text.isdecimal():
            raise ValueError("N must be a positive decimal integer (underscores are allowed).")
        target_n = int(n_text)
        if target_n <= 0:
            raise ValueError("N must be greater than zero.")
    except ValueError as exc:
        st.error(str(exc))
        return

    bit_count = st.session_state.last_bit_count
    with st.spinner("Generating primes and prime pairs…"):
        if bit_count <= SIEVE_MAX_BITS:
            primes = sieve_primes_in_range(bit_count)
            prime_mode = "Exhaustive prime enumeration"
        else:
            primes = generate_probable_primes(bit_count, st.session_state.last_prime_pool_size,
                                              st.session_state.last_seed)
            prime_mode = "Sampled probable-prime pool (Miller–Rabin, 16 rounds)"
        total, exact, p_values, q_values = make_pairs(
            primes, st.session_state.last_sample_limit, st.session_state.last_seed)

    lower = 1 << (bit_count - 1)
    shown = len(p_values)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bit count", f"{bit_count} bits")
    c2.metric("Primes generated", f"{len(primes):,}")
    c3.metric("Possible prime pairs", f"{total:,}")
    c4.metric("Pairs plotted", f"{shown:,}")
    st.caption(prime_mode)
    if bit_count > SIEVE_MAX_BITS:
        st.info("Large-number mode uses randomly generated probable primes, not a proof of primality. "
                "The Miller–Rabin test uses 16 rounds; the pool is reproducible from the seed.")
    if exact:
        st.success(f"All {total:,} unordered pairs from the generated prime pool are plotted.")
    else:
        st.warning(f"Sampled view: {shown:,} of {total:,} possible pairs from the prime pool are plotted.")

    possible_min, possible_max = lower * lower, ((1 << bit_count) - 1) ** 2
    if not (possible_min <= target_n <= possible_max):
        st.warning("N is outside the product range for two numbers of this bit length, "
                   "so the ideal hyperbola does not cross the plotted square.")
    st.plotly_chart(build_plot(p_values, q_values, lower, bit_count, target_n), use_container_width=True)
    st.markdown("**Reading the colours:** `log₂(pq / N) = 0` means the prime-pair product equals N; "
                "negative values mean `pq < N`; positive values mean `pq > N`. Hover over a point "
                "to see p, q, and their exact product.")
    with st.expander("How are the prime pairs generated?"):
        st.latex(r"m = \pi(2^b-1)-\pi(2^{b-1}-1), \qquad \#\text{unordered pairs}=\frac{m(m+1)}{2}")
        st.write("Pairs are unordered because p × q = q × p. For bit counts up to 24, "
                 "the app enumerates every prime in the range. Above 24 bits, it generates "
                 "a seeded pool of probable primes and plots pairs from that pool.")


if __name__ == "__main__":
    main()
