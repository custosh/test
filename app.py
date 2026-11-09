from __future__ import annotations

import math
import random
import time
from bisect import bisect_left, bisect_right

import numpy as np
import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "Prime-pair hyperbola explorer"
DEFAULT_SAMPLE_LIMIT = 100_000
SIEVE_MAX_BITS = 24
SMALL_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)

MODE_NEAR = "Near the hyperbola (pairs with pq ≈ N)"
MODE_SQUARE = "Whole prime square (all pairs)"


# ----------------------------------------------------------------------
# Primes
# ----------------------------------------------------------------------

def sieve_primes_in_range(bit_count: int) -> list[int]:
    """Return every prime with exactly bit_count bits (sorted)."""
    lower = 1 << (bit_count - 1)
    upper = (1 << bit_count) - 1

    flags = np.ones(upper - lower + 1, dtype=np.bool_)

    for p in range(2, math.isqrt(upper) + 1):
        first = max(p * p, ((lower + p - 1) // p) * p)
        flags[first - lower :: p] = False

    return (np.flatnonzero(flags) + lower).astype(object).tolist()


def is_probable_prime(n: int, rounds: int, rng: random.Random) -> bool:
    """Miller-Rabin probable-prime test."""
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
    rng = random.Random(seed)

    lower = 1 << (bit_count - 1)
    upper = (1 << bit_count) - 1

    found = set()

    while len(found) < count:
        candidate = rng.getrandbits(bit_count) | lower | 1

        if candidate > upper or candidate in found:
            continue

        if is_probable_prime(candidate, 16, rng):
            found.add(candidate)

    return sorted(found)


# ----------------------------------------------------------------------
# Pair selection
# ----------------------------------------------------------------------

def make_pairs_square(primes: list[int], sample_limit: int, seed: int):
    """Ordered pairs (p, q) drawn uniformly from the full prime square."""
    m = len(primes)
    total = m * m

    rng = random.Random(seed)
    count = min(total, sample_limit)
    indices = rng.sample(range(total), count)

    p_values = [primes[k // m] for k in indices]
    q_values = [primes[k % m] for k in indices]

    return total, count == total, p_values, q_values


def make_pairs_near_curve(
    primes: list[int],
    target_n: int,
    sample_limit: int,
    band: int,
    seed: int,
):
    """
    Ordered pairs (p, q) lying close to the curve pq = N.

    For every prime p, the `band` primes just below and just above N / p
    are taken as partners q. Because every p is visited, both (p, q) and
    (q, p) appear, so the cloud is symmetric about p = q and hugs the
    hyperbola along its whole length.
    """
    if not primes:
        return 0, True, [], []

    # Only p whose ideal partner N/p lies inside the prime range matter.
    p_min = -(-target_n // primes[-1])  # ceil(N / largest prime)
    p_max = target_n // primes[0]

    i_lo = bisect_left(primes, p_min)
    i_hi = bisect_right(primes, p_max)

    eligible = range(i_lo, i_hi)
    per_p = 2 * band
    wanted_p = max(1, sample_limit // per_p)

    rng = random.Random(seed)

    if len(eligible) <= wanted_p:
        chosen = eligible
        exact = True
    else:
        chosen = rng.sample(eligible, wanted_p)
        exact = False

    m = len(primes)
    p_values: list[int] = []
    q_values: list[int] = []

    for i in chosen:
        p = primes[i]
        center = bisect_left(primes, target_n // p)

        for j in range(max(0, center - band), min(m, center + band)):
            p_values.append(p)
            q_values.append(primes[j])

    total = len(p_values) if exact else len(eligible) * per_p

    return total, exact, p_values, q_values


# ----------------------------------------------------------------------
# Factorization
# ----------------------------------------------------------------------

_TRIAL_PRIMES = [
    p for p in range(2, 1000)
    if all(p % d for d in range(2, math.isqrt(p) + 1))
]


def pollard_brent(n: int, deadline: float, rng: random.Random):
    """Pollard-Brent rho. Returns a non-trivial divisor, or None on timeout."""
    if n % 2 == 0:
        return 2

    batch = 128

    while True:
        if time.monotonic() > deadline:
            return None

        y = rng.randrange(1, n)
        c = rng.randrange(1, n)
        g = r = q = 1
        x = ys = y

        while g == 1:
            x = y

            for i in range(r):
                y = (y * y + c) % n
                if (i & 1023) == 1023 and time.monotonic() > deadline:
                    return None

            k = 0
            while k < r and g == 1:
                ys = y

                for _ in range(min(batch, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n

                g = math.gcd(q, n)
                k += batch

                if g == 1 and time.monotonic() > deadline:
                    return None

            r *= 2

        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)

        if g != n:
            return g


@st.cache_data(show_spinner=False)
def factorize(n_text: str, time_limit: float):
    """
    Factorize N: trial division, Miller-Rabin and Pollard-Brent rho.

    Returns (prime_factors, unresolved, elapsed) where prime_factors is a
    sorted list of (prime, exponent) and unresolved lists composite
    cofactors that were not split before the time limit.
    """
    n = int(n_text)
    start = time.monotonic()
    deadline = start + time_limit
    rng = random.Random(0xFAC70)

    found: dict[int, int] = {}
    unresolved: list[int] = []

    for p in _TRIAL_PRIMES:
        while n % p == 0:
            found[p] = found.get(p, 0) + 1
            n //= p

    stack = [n] if n > 1 else []

    while stack:
        m = stack.pop()

        if m == 1:
            continue

        if is_probable_prime(m, 24, rng):
            found[m] = found.get(m, 0) + 1
            continue

        d = pollard_brent(m, deadline, rng)

        if d is None:
            unresolved.append(m)
            continue

        stack.extend((d, m // d))

    return sorted(found.items()), unresolved, time.monotonic() - start


def format_factorization(factors, unresolved) -> str:
    parts = [f"{p}^{e}" if e > 1 else str(p) for p, e in factors]
    parts += [f"[composite {m}]" for m in unresolved]
    return " × ".join(parts) if parts else "1"


def divisor_pairs_in_square(
    factors,
    unresolved,
    n: int,
    bit_count: int,
    max_divisors: int = 200_000,
    max_pairs: int = 50,
):
    """
    Pairs (d, N/d), d <= N/d, where both entries have exactly bit_count bits,
    i.e. the factorizations that land inside the plotted prime square.
    """
    items = list(factors) + [(m, 1) for m in unresolved]

    count = 1
    for _, e in items:
        count *= e + 1
    if count > max_divisors:
        return []

    divisors = [1]
    for p, e in items:
        divisors = [d * p**k for d in divisors for k in range(e + 1)]

    lower = 1 << (bit_count - 1)
    upper = (1 << bit_count) - 1

    pairs = set()
    for d in divisors:
        if lower <= d <= upper:
            q = n // d
            if lower <= q <= upper and d <= q:
                pairs.add((d, q))

    return sorted(pairs, key=lambda t: t[1] - t[0])[:max_pairs]


# ----------------------------------------------------------------------
# Numeric helpers
# ----------------------------------------------------------------------

def scaled_integer(value: int, exponent: int) -> float:
    """value / 2**exponent as a float, keeping only the top 53 bits."""
    if value == 0:
        return 0.0

    shift = max(0, value.bit_length() - 53)
    mantissa = float(value >> shift)

    return math.ldexp(mantissa, shift - exponent)


def make_normalizer(target_n: int):
    """
    Return f(v) = v / sqrt(N), safe for arbitrarily large integers.

    In these coordinates the ideal hyperbola is x * y = 1 and the
    balanced point (sqrt N, sqrt N) is (1, 1).
    """
    root = math.isqrt(target_n)
    e = max(root.bit_length() - 1, 0)
    r = scaled_integer(root, e)

    def norm(v: int) -> float:
        return scaled_integer(v, e) / r

    return norm


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def build_plot(
    p_values,
    q_values,
    bit_count,
    target_n,
    *,
    show_hyperbola=True,
    show_diagonal=True,
    show_n_pointer=True,
    known_pair=None,
    factor_pairs=None,
):
    norm = make_normalizer(target_n)

    lower = 1 << (bit_count - 1)
    upper = (1 << bit_count) - 1

    # Square that contains every possible prime pair, in sqrt(N) units.
    axis_min = norm(lower) * 0.97
    axis_max = norm(upper) * 1.03

    x = np.asarray([norm(p) for p in p_values], dtype=float)
    y = np.asarray([norm(q) for q in q_values], dtype=float)

    log_target = math.log2(target_n)
    log_error = np.asarray([
        math.log2(p) + math.log2(q) - log_target
        for p, q in zip(p_values, q_values)
    ], dtype=float)

    n_pts = len(p_values)
    customdata = np.empty((n_pts, 4), dtype=object)
    for k, (p, q) in enumerate(zip(p_values, q_values)):
        customdata[k, 0] = str(p)
        customdata[k, 1] = str(q)
        customdata[k, 2] = str(p * q)
        customdata[k, 3] = float(log_error[k])

    max_error = max(float(np.max(np.abs(log_error))), 1e-12) if n_pts else 1.0

    fig = go.Figure()

    # Prime pairs -----------------------------------------------------
    fig.add_trace(go.Scattergl(
        x=x,
        y=y,
        mode="markers",
        name="Prime pairs",
        customdata=customdata,
        marker=dict(
            size=5,
            opacity=0.8,
            color=log_error,
            colorscale="RdBu_r",
            cmid=0,
            cmin=-max_error,
            cmax=max_error,
            colorbar=dict(title="log₂(pq/N)"),
        ),
        hovertemplate=(
            "p = %{customdata[0]}<br>"
            "q = %{customdata[1]}<br>"
            "Product = %{customdata[2]}<br>"
            "Distance = %{customdata[3]:+.6f}"
            "<extra></extra>"
        ),
    ))

    # Ideal hyperbola x * y = C (C ≈ 1) -------------------------------
    C = norm(math.isqrt(target_n)) ** 2  # ≈ 1, robust for huge N

    if show_hyperbola and C > 0:
        x_start = max(axis_min, C / axis_max)
        x_end = min(axis_max, C / axis_min)

        if x_start < x_end:
            curve_x = np.geomspace(x_start, x_end, 1200)
            curve_y = C / curve_x

            fig.add_trace(go.Scatter(
                x=curve_x,
                y=curve_y,
                mode="lines",
                name="Ideal hyperbola pq = N",
                line=dict(color="#111827", width=3),
                hovertemplate="Ideal hyperbola<br>x × y = 1<extra></extra>",
            ))

    # Diagonal p = q --------------------------------------------------
    if show_diagonal:
        fig.add_trace(go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            name="p = q",
            line=dict(color="#64748b", width=1.5, dash="dot"),
            hoverinfo="skip",
        ))

    # Balanced reference (sqrt N, sqrt N) -----------------------------
    center = math.sqrt(C)

    if show_n_pointer and axis_min <= center <= axis_max:
        fig.add_trace(go.Scatter(
            x=[center],
            y=[center],
            mode="markers",
            name="√N reference",
            marker=dict(
                symbol="diamond",
                size=15,
                color="#f59e0b",
                line=dict(color="#78350f", width=2),
            ),
            hovertemplate="Balanced reference<br>p = q = √N<extra></extra>",
        ))

        fig.add_annotation(
            x=center,
            y=center,
            text="√N reference",
            showarrow=True,
            arrowhead=2,
            ax=65,
            ay=-45,
            bgcolor="white",
            bordercolor="#f59e0b",
            borderwidth=1,
        )

    # Known factors ---------------------------------------------------
    if known_pair is not None:
        kp, kq = known_pair
        positions = [(kp, kq)]

        if kp != kq:
            positions.append((kq, kp))

        fig.add_trace(go.Scatter(
            x=[norm(p) for p, _ in positions],
            y=[norm(q) for _, q in positions],
            mode="markers",
            name="Known factor pair",
            marker=dict(
                symbol="star",
                size=19,
                color="#16a34a",
                line=dict(color="#14532d", width=2),
            ),
            hovertemplate="Known factor pair<extra></extra>",
        ))

    # Factorization result --------------------------------------------
    if factor_pairs:
        positions = []
        for fp, fq in factor_pairs:
            positions.append((fp, fq))
            if fp != fq:
                positions.append((fq, fp))

        fig.add_trace(go.Scatter(
            x=[norm(p) for p, _ in positions],
            y=[norm(q) for _, q in positions],
            mode="markers",
            name="Factorization result",
            customdata=[[str(p), str(q)] for p, q in positions],
            marker=dict(
                symbol="circle",
                size=17,
                color="#22c55e",
                line=dict(color="#14532d", width=3),
            ),
            hovertemplate=(
                "Factorization of N<br>"
                "p = %{customdata[0]}<br>"
                "q = %{customdata[1]}"
                "<extra></extra>"
            ),
        ))

    # Layout ----------------------------------------------------------
    fig.update_layout(
        template="plotly_white",
        height=800,
        title=dict(text="Prime Products — Rectangular Hyperbola", x=0.02),
        xaxis=dict(
            title="p / √N",
            range=[axis_min, axis_max],
            showgrid=True,
            gridcolor="#e5e7eb",
        ),
        yaxis=dict(
            title="q / √N",
            range=[axis_min, axis_max],
            showgrid=True,
            gridcolor="#e5e7eb",
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(orientation="h", y=1.04, x=0),
        hovermode="closest",
        margin=dict(l=80, r=60, t=110, b=80),
    )

    return fig


# ----------------------------------------------------------------------
# Input parsing
# ----------------------------------------------------------------------

def parse_known_pair(p_text: str, q_text: str, target_n: int, bit_count: int):
    p_text = p_text.strip().replace("_", "")
    q_text = q_text.strip().replace("_", "")

    if not p_text and not q_text:
        return None, None

    if not p_text or not q_text:
        return None, "Enter both known factors, or leave both fields empty."

    if not p_text.isdecimal() or not q_text.isdecimal():
        return None, "Known factors must be positive decimal integers."

    p = int(p_text)
    q = int(q_text)

    if p <= 1 or q <= 1:
        return None, "Known factors must be greater than 1."

    lower = 1 << (bit_count - 1)
    upper = (1 << bit_count) - 1

    if not (lower <= p <= upper and lower <= q <= upper):
        return None, "Both known factors must have the selected bit count."

    if p * q != target_n:
        return None, "The known factors must multiply exactly to N."

    check_rng = random.Random(0xC0FFEE)

    if not is_probable_prime(p, 24, check_rng) or not is_probable_prime(q, 24, check_rng):
        return None, "Both known factors must pass the Miller–Rabin primality check."

    return (min(p, q), max(p, q)), None


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.write(
        "Each dot is a prime pair (p, q). Coordinates are the factor values "
        "divided by √N, so the target relation pq = N is the curve xy = 1. "
        "Colour shows how far pq is above or below N."
    )

    with st.sidebar:
        st.header("Parameters")

        bit_count = int(st.number_input(
            "Bit count per prime",
            min_value=2,
            max_value=4096,
            value=8,
            step=1,
        ))

        # Balanced default: the hyperbola crosses the whole prime square
        # from corner to corner instead of only touching one corner.
        suggested_n = 1 << (2 * bit_count - 1)

        st.caption(
            f"Prime range: 2^{bit_count - 1} to 2^{bit_count} − 1. "
            f"Balanced N: 2^{2 * bit_count - 1}"
        )

        if "target_n_text" not in st.session_state:
            st.session_state.target_n_text = str(suggested_n)

        if "previous_suggested_n" not in st.session_state:
            st.session_state.previous_suggested_n = suggested_n

        elif st.session_state.previous_suggested_n != suggested_n:
            if st.session_state.target_n_text == str(st.session_state.previous_suggested_n):
                st.session_state.target_n_text = str(suggested_n)

            st.session_state.previous_suggested_n = suggested_n

        if st.button("Reset N to the balanced default", use_container_width=True):
            st.session_state.target_n_text = str(suggested_n)

        target_text = st.text_input(
            "Target N (positive decimal integer)",
            key="target_n_text",
        )

        with st.expander("Known factors (optional)", expanded=False):
            known_p_text = st.text_input(
                "Known p", key="known_p_text", placeholder="Leave empty"
            )
            known_q_text = st.text_input(
                "Known q", key="known_q_text", placeholder="Leave empty"
            )
            st.caption(
                "When both are entered and p × q = N, "
                "the pair is highlighted with a green star."
            )

        st.divider()

        st.subheader("Pair selection")

        pair_mode = st.radio(
            "Which pairs to plot",
            [MODE_NEAR, MODE_SQUARE],
            label_visibility="collapsed",
        )

        band = 3
        if pair_mode == MODE_NEAR:
            band = int(st.slider(
                "Partners per side of the curve",
                1, 20, 3, 1,
                help="For each p, this many primes just below and just "
                     "above N/p are used as q. Larger = thicker band.",
            ))

        st.divider()

        st.subheader("Graph overlays")

        show_hyperbola = st.checkbox("Show ideal hyperbola pq = N", value=True)
        show_n_pointer = st.checkbox("Show N pointer at (√N, √N)", value=True)
        show_diagonal = st.checkbox("Show diagonal p = q", value=True)
        show_known_pair = st.checkbox("Show known factor marker", value=True)

        st.divider()

        st.subheader("Factorization")

        run_factorization = st.checkbox(
            "Factorize N and mark the result (green dot)",
            value=True,
        )

        factor_time_limit = st.slider(
            "Factorization time limit (seconds)", 1, 120, 10, 1
        )

        if bit_count <= SIEVE_MAX_BITS:
            st.caption("All primes in this bit range will be enumerated with a sieve.")
        else:
            prime_pool_size = st.slider(
                "Number of probable primes to generate", 100, 5000, 500, 100
            )
            st.caption(
                "For large bit counts, a reproducible pool is generated "
                "using Miller–Rabin."
            )

        sample_limit = st.slider(
            "Maximum plotted pairs",
            10_000, 250_000, DEFAULT_SAMPLE_LIMIT, 10_000,
        )

        seed = st.number_input("Generation seed", min_value=0, value=42, step=1)

        generate = st.button("Generate graph", type="primary", use_container_width=True)

    required_keys = (
        "last_bit_count",
        "last_target_text",
        "last_sample_limit",
        "last_seed",
        "last_pair_mode",
        "last_band",
        "last_prime_pool_size",
        "last_factorize",
        "last_factor_time",
    )

    # Session state survives code reloads, so a session started with an
    # older version of the app may be missing keys added later.
    if "has_generated" not in st.session_state or not all(
        k in st.session_state for k in required_keys
    ):
        st.session_state.has_generated = False

    if generate:
        st.session_state.has_generated = True
        st.session_state.last_bit_count = bit_count
        st.session_state.last_target_text = target_text
        st.session_state.last_sample_limit = int(sample_limit)
        st.session_state.last_seed = int(seed)
        st.session_state.last_pair_mode = pair_mode
        st.session_state.last_band = band
        st.session_state.last_factorize = bool(run_factorization)
        st.session_state.last_factor_time = int(factor_time_limit)
        st.session_state.last_prime_pool_size = (
            int(prime_pool_size) if bit_count > SIEVE_MAX_BITS else None
        )

    if not st.session_state.has_generated:
        st.info(
            "Choose the bit count and N in the sidebar, "
            "then select Generate graph."
        )
        return

    try:
        n_text = st.session_state.last_target_text.strip().replace("_", "")

        if not n_text or not n_text.isdecimal():
            raise ValueError(
                "N must be a positive decimal integer (underscores are allowed)."
            )

        target_n = int(n_text)

        if target_n <= 0:
            raise ValueError("N must be greater than zero.")

    except ValueError as exc:
        st.error(str(exc))
        return

    bit_count = st.session_state.last_bit_count
    last_mode = st.session_state.last_pair_mode

    with st.spinner("Generating primes and prime pairs…"):
        if bit_count <= SIEVE_MAX_BITS:
            primes = sieve_primes_in_range(bit_count)
            prime_mode = "Exhaustive prime enumeration"
        else:
            primes = generate_probable_primes(
                bit_count,
                st.session_state.last_prime_pool_size,
                st.session_state.last_seed,
            )
            prime_mode = "Sampled probable-prime pool (Miller–Rabin, 16 rounds)"

        if last_mode == MODE_NEAR:
            total, exact, p_values, q_values = make_pairs_near_curve(
                primes,
                target_n,
                st.session_state.last_sample_limit,
                st.session_state.last_band,
                st.session_state.last_seed,
            )
        else:
            total, exact, p_values, q_values = make_pairs_square(
                primes,
                st.session_state.last_sample_limit,
                st.session_state.last_seed,
            )

    lower = 1 << (bit_count - 1)
    shown = len(p_values)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bit count", f"{bit_count} bits")
    c2.metric("Primes generated", f"{len(primes):,}")
    c3.metric(
        "Candidate pairs" if last_mode == MODE_NEAR else "Possible ordered pairs",
        f"{total:,}",
    )
    c4.metric("Pairs plotted", f"{shown:,}")

    st.caption(prime_mode)

    if bit_count > SIEVE_MAX_BITS:
        st.info(
            "Large-number mode uses randomly generated probable primes, "
            "not a proof of primality. The Miller–Rabin test uses 16 rounds; "
            "the pool is reproducible from the seed."
        )

    possible_min = lower * lower
    possible_max = ((1 << bit_count) - 1) ** 2
    n_in_range = possible_min <= target_n <= possible_max

    if not n_in_range:
        st.warning(
            "N is outside the product range for two numbers of this bit "
            "length, so the ideal hyperbola does not cross the prime square."
        )

    if shown == 0:
        st.warning(
            "No prime pairs lie near the hyperbola for this N. "
            "Try a different N or the whole-square mode."
        )
    elif exact:
        st.success(f"All {shown:,} selected pairs are plotted.")
    else:
        st.warning(f"Sampled view: {shown:,} of about {total:,} pairs are plotted.")

    known_pair, known_error = parse_known_pair(
        known_p_text, known_q_text, target_n, bit_count
    )

    if known_error:
        st.warning(known_error)
    elif known_pair is not None:
        st.success(
            "Known factor pair validated: p × q = N. "
            "It is highlighted with a green star."
        )

    factor_pairs = None

    if st.session_state.last_factorize:
        with st.spinner("Factorizing N…"):
            factors, unresolved, elapsed = factorize(
                str(target_n),
                float(st.session_state.last_factor_time),
            )

        expression = format_factorization(factors, unresolved)

        if not unresolved:
            st.success(f"Factorized in {elapsed:.2f}s: N = {expression}")
        else:
            st.warning(
                f"Time limit reached after {elapsed:.2f}s; "
                f"partial factorization: N = {expression}"
            )

        factor_pairs = divisor_pairs_in_square(
            factors, unresolved, target_n, bit_count
        )

        if not factor_pairs:
            st.info(
                "No factor pair of N has both factors with the selected "
                "bit count, so there is nothing to mark inside the plotted "
                "square."
            )

    st.plotly_chart(
        build_plot(
            p_values,
            q_values,
            bit_count,
            target_n,
            show_hyperbola=show_hyperbola,
            show_diagonal=show_diagonal,
            show_n_pointer=show_n_pointer,
            known_pair=known_pair if show_known_pair else None,
            factor_pairs=factor_pairs,
        ),
        use_container_width=True,
    )

    st.markdown(
        "**Reading the graph:** each point sits at its actual `(p, q) / √N` "
        "position, with no jitter. The black curve is `pq = N` (`xy = 1`). "
        "Colour shows `log₂(pq / N)`: zero is an exact match, negative is "
        "below N, positive is above N. The orange diamond marks `(√N, √N)`; "
        "the optional green star marks the known factor pair. Hover over a "
        "dot to inspect its exact p, q and product."
    )

    with st.expander("How are the prime pairs generated?"):
        st.latex(
            r"m = \pi(2^b-1)-\pi(2^{b-1}-1), "
            r"\qquad "
            r"\#\text{ordered pairs}=m^2"
        )

        st.write(
            "Pairs are ordered, so both (p, q) and (q, p) appear and the "
            "plot is symmetric about p = q. In 'Near the hyperbola' mode, "
            "for every prime p the primes closest to N / p (a few on each "
            "side) are chosen as q, so the points follow the curve pq = N "
            "along its whole length. In 'Whole prime square' mode pairs are "
            "sampled uniformly from all m² combinations, which fills the "
            "square [2^(b−1), 2^b)². Up to 24 bits every prime in the range "
            "is enumerated; above that a seeded pool of probable primes is used."
        )


if __name__ == "__main__":
    main()
