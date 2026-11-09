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
    """Return every prime with exactly bit_count bits."""
    lower = 1 << (bit_count - 1)
    upper = (1 << bit_count) - 1

    flags = np.ones(upper - lower + 1, dtype=np.bool_)

    for p in range(2, math.isqrt(upper) + 1):
        if p * p > upper:
            break

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


def generate_probable_primes(
    bit_count: int,
    count: int,
    seed: int
) -> list[int]:

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


def row_start(i: int, m: int) -> int:
    return i * m - (i * (i - 1)) // 2


def unrank_pair(k: int, m: int) -> tuple[int, int]:
    """Convert an index into an unordered pair index."""

    discriminant = (2 * m + 1) ** 2 - 8 * k

    i = ((2 * m + 1) - math.isqrt(discriminant)) // 2

    while i > 0 and row_start(i, m) > k:
        i -= 1

    while i + 1 < m and row_start(i + 1, m) <= k:
        i += 1

    return i, i + (k - row_start(i, m))


def sampled_indices(
    total: int,
    count: int,
    seed: int
) -> list[int]:

    rng = random.Random(seed)

    if count >= total:
        return list(range(total))

    return rng.sample(range(total), count)


def make_pairs(
    primes: list[int],
    sample_limit: int,
    seed: int
):
    """
    Generate ordered prime pairs.

    Unlike the previous implementation, this does not
    enforce p <= q. Both (p,q) and (q,p) are represented.
    """

    m = len(primes)
    total = m * m

    rng = random.Random(seed)
    count = min(total, sample_limit)

    # Randomize pair selection rather than using triangular indexing.
    indices = rng.sample(range(total), count)

    p_values = []
    q_values = []

    for k in indices:
        i = k // m
        j = k % m

        p_values.append(primes[i])
        q_values.append(primes[j])

    exact = count == total

    return total, exact, p_values, q_values


def scaled_integer(value: int, exponent: int) -> float:
    """
    Convert a potentially enormous integer into a
    floating-point value divided by 2**exponent.

    Retains the most significant bits to avoid overflow.
    """

    if value == 0:
        return 0.0

    shift = max(0, value.bit_length() - 53)
    mantissa = float(value >> shift)

    return math.ldexp(mantissa, shift - exponent)


def build_plot(
    p_values,
    q_values,
    lower,
    bit_count,
    target_n,
    *,
    show_hyperbola=True,
    show_diagonal=True,
    show_n_pointer=True,
    known_pair=None,
    jitter=0.004,
    seed=42,
):

    exponent = bit_count - 1
    axis_max = 2.05

    # Normalize coordinates to avoid floating-point overflow.
    x = np.asarray([
        scaled_integer(p, exponent)
        for p in p_values
    ])

    y = np.asarray([
        scaled_integer(q, exponent)
        for q in q_values
    ])

    # Random visual displacement to reduce rigid grid patterns.
    rng = np.random.default_rng(seed)

    x_display = np.clip(
        x + rng.uniform(-jitter, jitter, len(x)),
        0,
        axis_max
    )

    y_display = np.clip(
        y + rng.uniform(-jitter, jitter, len(y)),
        0,
        axis_max
    )

    # Product error, expressed in bits.
    log_error = np.asarray([
        math.log2(p) + math.log2(q) - math.log2(target_n)
        for p, q in zip(p_values, q_values)
    ])

    products = [
        str(p * q)
        for p, q in zip(p_values, q_values)
    ]

    fig = go.Figure()

    # --------------------------------------------------
    # PRIME PAIRS
    # --------------------------------------------------

    max_error = max(
        float(np.max(np.abs(log_error))),
        1e-12
    ) if len(log_error) else 1.0

    fig.add_trace(go.Scattergl(
        x=x_display,
        y=y_display,

        mode="markers",
        name="Prime pairs",

        customdata=np.column_stack([
            [str(p) for p in p_values],
            [str(q) for q in q_values],
            products,
            log_error
        ]),

        marker=dict(
            size=5,
            opacity=0.8,

            color=log_error,
            colorscale="RdBu_r",
            cmid=0,
            cmin=-max_error,
            cmax=max_error,

            colorbar=dict(
                title="log₂(pq/N)"
            )
        ),

        hovertemplate=(
            "p = %{customdata[0]}<br>"
            "q = %{customdata[1]}<br>"
            "Product = %{customdata[2]}<br>"
            "Distance = %{customdata[3]:+.6f}"
            "<extra></extra>"
        )
    ))

    # --------------------------------------------------
    # RECTANGULAR HYPERBOLA
    # --------------------------------------------------

    # Normalized equation:
    #
    # x * y = N / 2**(2 * exponent)
    #
    # Therefore:
    #
    # y = C / x

    C = scaled_integer(
        target_n,
        2 * exponent
    )

    if show_hyperbola and 0 < C <= axis_max ** 2:

        # Start where the curve enters the upper boundary.
        x_start = max(C / axis_max, 1e-10)

        curve_x = np.geomspace(
            x_start,
            axis_max,
            1200
        )

        curve_y = C / curve_x

        visible = (
            (curve_x <= axis_max)
            & (curve_y <= axis_max)
        )

        fig.add_trace(go.Scatter(
            x=curve_x[visible],
            y=curve_y[visible],

            mode="lines",
            name="Ideal rectangular hyperbola",

            line=dict(
                color="#111827",
                width=3
            ),

            hovertemplate=(
                "Ideal hyperbola<br>"
                "x × y = C"
                "<extra></extra>"
            )
        ))

    # --------------------------------------------------
    # ASYMPTOTES: X AND Y AXES
    # --------------------------------------------------

    # The rectangular hyperbola has perpendicular
    # asymptotes at x = 0 and y = 0.
    #
    # These are represented by the graph axes.

    # --------------------------------------------------
    # DIAGONAL
    # --------------------------------------------------

    if show_diagonal:

        fig.add_trace(go.Scatter(
            x=[0, axis_max],
            y=[0, axis_max],

            mode="lines",
            name="p = q",

            line=dict(
                color="#64748b",
                width=1.5,
                dash="dot"
            ),

            hoverinfo="skip"
        ))

    # --------------------------------------------------
    # BALANCED N POINTER
    # --------------------------------------------------

    if show_n_pointer and 0 < C <= axis_max ** 2:

        center = math.sqrt(C)

        fig.add_trace(go.Scatter(
            x=[center],
            y=[center],

            mode="markers",
            name="N reference",

            marker=dict(
                symbol="diamond",
                size=15,
                color="#f59e0b",
                line=dict(
                    color="#78350f",
                    width=2
                )
            ),

            hovertemplate=(
                "Balanced reference<br>"
                "x = y = √C"
                "<extra></extra>"
            )
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
            borderwidth=1
        )

    # --------------------------------------------------
    # KNOWN FACTORS
    # --------------------------------------------------

    if known_pair is not None:

        kp, kq = known_pair

        positions = [(kp, kq)]

        if kp != kq:
            positions.append((kq, kp))

        known_x = [
            scaled_integer(p, exponent)
            for p, q in positions
        ]

        known_y = [
            scaled_integer(q, exponent)
            for p, q in positions
        ]

        fig.add_trace(go.Scatter(
            x=known_x,
            y=known_y,

            mode="markers",
            name="Known factor pair",

            marker=dict(
                symbol="star",
                size=19,
                color="#16a34a",
                line=dict(
                    color="#14532d",
                    width=2
                )
            ),

            hovertemplate=(
                "Known factor pair"
                "<extra></extra>"
            )
        ))

    # --------------------------------------------------
    # GRAPH LAYOUT
    # --------------------------------------------------

    fig.update_layout(
        template="plotly_white",
        height=800,

        title=dict(
            text="Prime Products — Rectangular Hyperbola",
            x=0.02
        ),

        xaxis=dict(
            title=f"p / 2^{exponent}",
            range=[0, axis_max],

            dtick=0.25,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="#111827",

            showgrid=True,
            gridcolor="#e5e7eb"
        ),

        yaxis=dict(
            title=f"q / 2^{exponent}",
            range=[0, axis_max],

            dtick=0.25,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="#111827",

            showgrid=True,
            gridcolor="#e5e7eb",

            scaleanchor="x",
            scaleratio=1
        ),

        legend=dict(
            orientation="h",
            y=1.04,
            x=0
        ),

        hovermode="closest",

        margin=dict(
            l=80,
            r=60,
            t=110,
            b=80
        )
    )

    return fig

def parse_known_pair(
    p_text: str,
    q_text: str,
    target_n: int,
    bit_count: int
):

    p_text = p_text.strip().replace("_", "")
    q_text = q_text.strip().replace("_", "")

    if not p_text and not q_text:
        return None, None

    if not p_text or not q_text:
        return None, (
            "Enter both known factors, or "
            "leave both fields empty."
        )

    if not p_text.isdecimal() or not q_text.isdecimal():
        return None, (
            "Known factors must be positive decimal integers."
        )

    p = int(p_text)
    q = int(q_text)

    if p <= 1 or q <= 1:
        return None, "Known factors must be greater than 1."

    lower = 1 << (bit_count - 1)
    upper = (1 << bit_count) - 1

    if not (
        lower <= p <= upper
        and lower <= q <= upper
    ):
        return None, (
            "Both known factors must have the selected bit count."
        )

    if p * q != target_n:
        return None, (
            "The known factors must multiply exactly to N."
        )

    check_rng = random.Random(0xC0FFEE)

    if (
        not is_probable_prime(p, 24, check_rng)
        or not is_probable_prime(q, 24, check_rng)
    ):
        return None, (
            "Both known factors must pass "
            "the Miller–Rabin primality check."
        )

    return (min(p, q), max(p, q)), None


def main():

    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide"
    )

    st.title(APP_TITLE)

    st.write(
        "Each dot is a prime pair (p, q). "
        "Its coordinates are the normalized factor values; "
        "its colour shows how far pq is above or below N. "
        "The dashed curve is the exact target relation pq = N."
    )

    with st.sidebar:

        st.header("Parameters")

        bit_count = st.number_input(
            "Bit count per prime",
            min_value=2,
            max_value=4096,
            value=8,
            step=1
        )

        bit_count = int(bit_count)

        suggested_n = 1 << (
            2 * (bit_count - 1)
        )

        st.caption(
            f"Prime range: 2^{bit_count - 1} "
            f"to 2^{bit_count} − 1. "
            f"Suggested N: 2^{2 * (bit_count - 1)}"
        )

        if "target_n_text" not in st.session_state:
            st.session_state.target_n_text = str(suggested_n)

        if "previous_suggested_n" not in st.session_state:
            st.session_state.previous_suggested_n = suggested_n

        elif (
            st.session_state.previous_suggested_n
            != suggested_n
        ):

            if (
                st.session_state.target_n_text
                == str(st.session_state.previous_suggested_n)
            ):
                st.session_state.target_n_text = str(suggested_n)

            st.session_state.previous_suggested_n = suggested_n

        if st.button(
            "Reset N to the minimum possible product",
            use_container_width=True
        ):
            st.session_state.target_n_text = str(suggested_n)

        target_text = st.text_input(
            "Target N (positive decimal integer)",
            key="target_n_text"
        )

        with st.expander(
            "Known factors (optional)",
            expanded=False
        ):

            known_p_text = st.text_input(
                "Known p",
                key="known_p_text",
                placeholder="Leave empty"
            )

            known_q_text = st.text_input(
                "Known q",
                key="known_q_text",
                placeholder="Leave empty"
            )

            st.caption(
                "When both are entered and p × q = N, "
                "the pair is highlighted with a green star."
            )

        st.divider()

        st.subheader("Graph overlays")

        show_hyperbola = st.checkbox(
            "Show ideal hyperbola pq = N",
            value=True
        )

        show_n_pointer = st.checkbox(
            "Show N pointer at (√N, √N)",
            value=True
        )

        show_diagonal = st.checkbox(
            "Show diagonal p = q",
            value=True
        )

        show_known_pair = st.checkbox(
            "Show known factor marker",
            value=True
        )

        if bit_count <= SIEVE_MAX_BITS:

            st.caption(
                "All primes in this bit range "
                "will be enumerated with a sieve."
            )

        else:

            prime_pool_size = st.slider(
                "Number of probable primes to generate",
                100,
                5000,
                500,
                100
            )

            st.caption(
                "For large bit counts, a reproducible pool "
                "is generated using Miller–Rabin."
            )

        sample_limit = st.slider(
            "Maximum plotted pairs when sampling",
            10_000,
            250_000,
            DEFAULT_SAMPLE_LIMIT,
            10_000
        )

        seed = st.number_input(
            "Generation seed",
            min_value=0,
            value=42,
            step=1
        )

        generate = st.button(
            "Generate graph",
            type="primary",
            use_container_width=True
        )

    if "has_generated" not in st.session_state:
        st.session_state.has_generated = False

    if generate:

        st.session_state.has_generated = True

        st.session_state.last_bit_count = bit_count
        st.session_state.last_target_text = target_text
        st.session_state.last_sample_limit = int(sample_limit)
        st.session_state.last_seed = int(seed)

        st.session_state.last_prime_pool_size = (
            int(prime_pool_size)
            if bit_count > SIEVE_MAX_BITS
            else None
        )

    if not st.session_state.has_generated:

        st.info(
            "Choose the bit count and N in the sidebar, "
            "then select Generate graph."
        )

        return

    try:

        n_text = (
            st.session_state.last_target_text
            .strip()
            .replace("_", "")
        )

        if not n_text or not n_text.isdecimal():
            raise ValueError(
                "N must be a positive decimal integer "
                "(underscores are allowed)."
            )

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

            primes = generate_probable_primes(
                bit_count,
                st.session_state.last_prime_pool_size,
                st.session_state.last_seed
            )

            prime_mode = (
                "Sampled probable-prime pool "
                "(Miller–Rabin, 16 rounds)"
            )

        total, exact, p_values, q_values = make_pairs(
            primes,
            st.session_state.last_sample_limit,
            st.session_state.last_seed
        )

    lower = 1 << (bit_count - 1)

    shown = len(p_values)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Bit count", f"{bit_count} bits")
    c2.metric("Primes generated", f"{len(primes):,}")
    c3.metric("Possible prime pairs", f"{total:,}")
    c4.metric("Pairs plotted", f"{shown:,}")

    st.caption(prime_mode)

    if bit_count > SIEVE_MAX_BITS:

        st.info(
            "Large-number mode uses randomly generated "
            "probable primes, not a proof of primality. "
            "The Miller–Rabin test uses 16 rounds; "
            "the pool is reproducible from the seed."
        )

    if exact:

        st.success(
            f"All {total:,} unordered pairs "
            "from the generated prime pool are plotted."
        )

    else:

        st.warning(
            f"Sampled view: {shown:,} of {total:,} "
            "possible pairs from the prime pool are plotted."
        )

    possible_min = lower * lower

    possible_max = (
        (1 << bit_count) - 1
    ) ** 2

    if not (
        possible_min <= target_n <= possible_max
    ):

        st.warning(
            "N is outside the product range for two "
            "numbers of this bit length, so the ideal "
            "hyperbola does not cross the plotted square."
        )

    known_pair, known_error = parse_known_pair(
        known_p_text,
        known_q_text,
        target_n,
        bit_count
    )

    if known_error:

        st.warning(known_error)

    elif known_pair is not None:

        st.success(
            "Known factor pair validated: p × q = N. "
            "It is highlighted with a green star."
        )

    st.plotly_chart(
        build_plot(
            p_values,
            q_values,
            lower,
            bit_count,
            target_n,
            show_hyperbola=show_hyperbola,
            show_diagonal=show_diagonal,
            show_n_pointer=show_n_pointer,
            known_pair=(
                known_pair
                if show_known_pair
                else None
            ),
        ),
        use_container_width=True
    )

    st.markdown(
        "**Reading the graph:** each point stays at its "
        "actual normalized `(p, q)` coordinates. "
        "The dashed curve is `pq = N`; points on it have "
        "an exact product match. Colour shows "
        "`log₂(pq / N)`: zero is an exact match, "
        "negative is below N, and positive is above N. "
        "The orange diamond marks the balanced reference "
        "`(√N, √N)`; the optional green star marks the "
        "known factor pair. Hover over a dot to inspect "
        "its exact p, q and product."
    )

    with st.expander("How are the prime pairs generated?"):

        st.latex(
            r"m = \pi(2^b-1)-\pi(2^{b-1}-1), "
            r"\qquad "
            r"\#\text{unordered pairs}=\frac{m(m+1)}{2}"
        )

        st.write(
            "Pairs are unordered because p × q = q × p. "
            "For bit counts up to 24, the app enumerates "
            "every prime in the range. Above 24 bits, "
            "it generates a seeded pool of probable primes "
            "and plots pairs from that pool."
        )


if __name__ == "__main__":
    main()
