import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics.pairwise import cosine_similarity

from matplotlib.dates import YearLocator, MonthLocator
from datetime import timedelta
from cycler import cycler
import matplotlib.colors as mcolors


def plot_lift(matrix, topic_names, ordered=False, logscale=False):
    fig = plt.figure()
    gs = GridSpec(1, 3, width_ratios=[10, 1, 0.4])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cbax = fig.add_subplot(gs[2])

    if ordered:
        order = np.argsort(np.mean(np.abs(matrix - 1), 0))[::-1]
        sorted_cooc = matrix[order].T[order]
    else:
        order = np.arange(len(topic_names))
        sorted_cooc = matrix

    if logscale:
        tmp_cooc = sorted_cooc.copy()
        tmp_cooc[np.eye(len(topic_names)) > 0] = np.nan
        log_cooc = np.log10(tmp_cooc)
        vlim = np.max(np.abs(log_cooc[~np.isnan(log_cooc)]))
        im = ax0.imshow(log_cooc, cmap="bwr", vmin=-vlim, vmax=vlim)

        log_cooc[np.eye(len(topic_names)) > 0] = np.mean(log_cooc[~np.isnan(log_cooc)])
        print(np.mean(log_cooc, 1))
        ax1.imshow(
            np.mean(log_cooc, 1, keepdims=True), cmap="bwr", vmin=-vlim, vmax=vlim
        )  # norm=norm)
    else:
        vmax = np.max(sorted_cooc[~np.isnan(sorted_cooc)])
        vmin = np.min(sorted_cooc[~np.isnan(sorted_cooc)])

        norm = TwoSlopeNorm(1, vmin, vmax)
        sorted_cooc[np.eye(len(topic_names)) > 0] = np.nan

        im = ax0.imshow(sorted_cooc, cmap="bwr", norm=norm)
        sorted_cooc[np.eye(len(topic_names)) > 0] = np.mean(
            sorted_cooc[~np.isnan(sorted_cooc)]
        )
        print(np.mean(sorted_cooc, 1))
        ax1.imshow(np.mean(sorted_cooc, 1, keepdims=True), cmap="bwr", norm=norm)

    pos = ax0.get_position()
    left, bottom, width, height = pos.x0, pos.y0, pos.width, pos.height

    ax1.tick_params(
        axis="both",
        which="both",
        bottom=True,
        left=False,
        labelbottom=True,
        labelleft=False,
    )
    ax1.set_xticks([0], ["Mean"], rotation=90)
    ax1.set_position((left + width, bottom, 0.1 * width, height))

    plt.colorbar(im, cbax)
    cbax.set_position((left + 1.15 * width, bottom, 0.05 * width, height))

    ax0.set_xticks(range(len(topic_names)), topic_names[order], rotation=90)
    ax0.set_yticks(range(len(topic_names)), topic_names[order])
    ax0.set_title("Normalized Deviation from Expected Co-Occurrence (Lift)")
    plt.show()


def plot_cramer(matrix, topic_names):
    matrix[np.eye(len(topic_names)) > 0] = np.nan
    im = plt.imshow(matrix, cmap="RdYlBu_r", vmin=0)

    plt.colorbar(im)

    plt.xticks(range(len(topic_names)), topic_names, rotation=90)
    plt.yticks(range(len(topic_names)), topic_names)
    plt.title("Cramér's V of Topic Pairs")
    plt.show()


def plot_scaled_lift(
    lift_matrix, cramer_matrix, topic_names, show_mean=False, offset=1
):
    fig = plt.figure()
    gs = GridSpec(1, 2 + show_mean, width_ratios=[10, 1, 0.4] if show_mean else [10, 1])
    ax0 = fig.add_subplot(gs[0])
    if show_mean:
        ax1 = fig.add_subplot(gs[1])
    cbax = fig.add_subplot(gs[1 + show_mean])

    lift_matrix[np.eye(len(lift_matrix)) > 0] = 1
    cooc_sign = (lift_matrix > 1) * 2 - 1
    scaled = cramer_matrix * np.abs(lift_matrix - 1) * cooc_sign + offset
    vmax = np.max(scaled[~np.isnan(scaled)])
    vmin = np.min(scaled[~np.isnan(scaled)])

    norm = TwoSlopeNorm(offset, vmin, vmax)

    im = ax0.imshow(scaled, cmap="bwr", norm=norm)

    plt.colorbar(im, cbax)

    pos = ax0.get_position()
    left, bottom, width, height = pos.x0, pos.y0, pos.width, pos.height

    if show_mean:
        scaled[np.eye(len(scaled)) > 0] = np.mean(scaled[~np.isnan(scaled)])
        ax1.imshow(np.mean(scaled, 1, keepdims=True), cmap="bwr", norm=norm)
        ax1.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=False,
            labelbottom=True,
            labelleft=False,
        )
        ax1.set_xticks([0], ["Mean"], rotation=90)
        ax1.set_position((left + width, bottom, 0.1 * width, height))

        cbax.set_position((left + 1.15 * width, bottom, 0.05 * width, height))
    else:
        cbax.set_position((left + width * 1.07, bottom, 0.05 * width, height))

    ax0.set_xticks(range(len(topic_names)), topic_names, rotation=90)
    ax0.set_yticks(range(len(topic_names)), topic_names)
    ax0.set_title("Topic Lift Scaled by Association Strength")

    plt.show()


def plot_similarity(topic_embeddings, topic_names, ordered=False, cmap="RdYlBu_r"):
    fig = plt.figure()
    gs = GridSpec(1, 3, width_ratios=[10, 1, 0.4])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cbax = fig.add_subplot(gs[2])

    sim_mat = np.array(cosine_similarity(topic_embeddings, topic_embeddings))

    # norm = TwoSlopeNorm(0.5, 0, 1)
    ticks = list(range(sim_mat.shape[0]))

    if ordered:
        order = np.argsort(np.mean(sim_mat, 0))[::-1]

        ordered_embed = [topic_embeddings[i] for i in order]
        ordered_names = [topic_names[i] for i in order]

        sim_mat_2 = np.array(cosine_similarity(ordered_embed, ordered_embed))
        sim_mat_2[np.eye(sim_mat_2.shape[0]).astype(bool)] = np.nan

        im = plt.imshow(sim_mat_2, cmap=cmap)
        plt.xticks(ticks, ordered_names, rotation=90)
        plt.yticks(ticks, ordered_names)
    else:
        sim_mat[np.eye(sim_mat.shape[0]).astype(bool)] = np.nan
        im = ax0.imshow(sim_mat, cmap=cmap)
        ax0.set_xticks(ticks, topic_names, rotation=90)
        ax0.set_yticks(ticks, topic_names)

        pos = ax0.get_position()
        left, bottom, width, height = pos.x0, pos.y0, pos.width, pos.height

        sim_mat[np.eye(sim_mat.shape[0]).astype(bool)] = np.mean(
            sim_mat[~np.isnan(sim_mat)]
        )
        vmin = np.min(sim_mat)
        vmax = np.max(sim_mat)
        ax1.imshow(np.mean(sim_mat, 1, keepdims=True), cmap=cmap, vmin=vmin, vmax=vmax)

        ax1.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=False,
            labelbottom=True,
            labelleft=False,
        )
        ax1.set_xticks([0], ["Mean"], rotation=90)
        ax1.set_position((left + width, bottom, 0.1 * width, height))

    plt.colorbar(im, cbax)
    cbax.set_position((left + 1.15 * width, bottom, 0.05 * width, height))

    ax0.set_title("Topic Embedding Cosine Similarity")
    plt.show()


def plot_topic_ratings(sentences_df, topic_names):
    topic_ratings = sentences_df[["topic", "rating"]].groupby("topic").mean()
    topic_ratings["sdev"] = (
        sentences_df[["topic", "rating"]].groupby("topic").std()["rating"].values
    )
    topic_ratings["name"] = topic_names
    topic_ratings.sort_values("rating", inplace=True)
    avg_rating = sentences_df["rating"].mean()

    hbars = plt.barh(
        topic_ratings["name"].values,
        topic_ratings["rating"].values - 1,
        left=1,
        color=(0, 0.7, 0.5),
    )
    plt.errorbar(
        topic_ratings["rating"].values,
        list(range(len(topic_ratings))),
        xerr=topic_ratings["sdev"].values,
        fmt="|",
        color="r",
        capsize=4,
        elinewidth=0,
        capthick=3,
    )
    plt.bar_label(hbars, [f" {v:.2f}" for v in topic_ratings["rating"].values])
    plt.gcf().set_size_inches(5, 6)
    plt.vlines(
        avg_rating, -1.5, len(topic_ratings) - 0.5, colors="0.8", linestyles="dashed"
    )
    plt.vlines(5, -2, len(topic_ratings), colors="0.8")
    plt.text(avg_rating - 0.1, -1.5, f"Avg: {avg_rating:.2f}", ha="right")
    plt.xlabel("Topic Avg Rating")
    plt.gca().set_xlim(1, 6)
    plt.gca().set_ylim(-2, 26)
    plt.suptitle("Average Review Rating per Topic", y=0.95)
    plt.title("with Std. Deviation Error Bars", fontsize=10)

    plt.show()


def print_reviews(review_df, topic_names, N=20):
    # review_df.sort_values('topic_probs', ascending=False, inplace=True)

    topic_ratings = review_df[["topic", "rating"]].groupby("topic").mean()
    topic_ratings["sdev"] = (
        review_df[["topic", "rating"]].groupby("topic").std()["rating"].values
    )
    topic_ratings["name"] = topic_names
    topic_ratings.sort_index(inplace=True)

    unique_topics = review_df.sort_values("topic")["topic"].unique()
    names = topic_ratings["name"].values
    ratings = topic_ratings["rating"].values

    for u in unique_topics[1:]:
        subset = review_df.loc[
            review_df["topic"] == u, ["title_product", "sentences"]
        ].head(N)
        print(f"\nTopic: {names[u+1]} ({ratings[u+1]:.2f})")
        for i in range(len(subset)):
            print(
                "\n".join(
                    textwrap.wrap(
                        subset["sentences"].values[i],
                        200,
                        initial_indent="  > ",
                        subsequent_indent="    ",
                    )
                )
            )


def _EMA(values, length=10):
    # Assume values is a numpy array
    out = np.zeros_like(values)
    out[0, :] = values[0, :]

    for i, col in enumerate(values[1:, ...]):
        out[i, :] = out[i - 1, :] + (col - out[i - 1, :]) / length

    return out


def plot_topics_over_time(sentences_df, topic_names):
    minTime = min(sentences_df["datetime"])
    maxTime = max(sentences_df["datetime"])
    N = 50
    bins = [
        minTime + timedelta(seconds=t)
        for t in np.linspace(0, (maxTime - minTime).total_seconds(), N)
    ]
    T = len(np.unique(topic_names)) - 1
    bin_idxs = pd.cut(sentences_df["datetime"], bins, labels=False).values
    counts = np.zeros((N, T))
    for i in range(N):
        subset = bin_idxs == i
        for j in range(T):
            counts[i, j] = sum(sentences_df.loc[subset, "topic"] == j)

    smooth_counts = _EMA(counts, 3)

    fig, ax = plt.subplots()
    axs = fig.axes[0]
    fig.set_size_inches(10, 6)

    cyc = cycler(marker=list(".sx^")) * cycler(
        color=list(mcolors.TABLEAU_COLORS.keys())
    )
    axs.set_prop_cycle(cyc)

    cts = smooth_counts
    normd_counts = cts / np.sum(cts, 0)
    avg_counts = np.mean(normd_counts, 1)

    plt.plot(bins, normd_counts, markersize=5)

    axs.xaxis.set_major_locator(YearLocator())
    axs.xaxis.set_minor_locator(MonthLocator([7]))

    plt.plot(bins, avg_counts, "k--", linewidth=3)
    plt.legend(
        topic_names[1:].tolist() + ["Avg"],
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    plt.xticks(rotation=45)
    plt.grid(which="minor", axis="x")
    plt.grid()

    plt.xlabel("Date")
    plt.ylabel("Proportion of Total Count per Topic")
    plt.title("Topic Mentions Over Time (Binned, Smoothed)")

    plt.tight_layout()
    plt.show()
