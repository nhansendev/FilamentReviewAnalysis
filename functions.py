import re
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from constants import DATA_DIR


def get_data_filepath(file):
    return os.path.join(DATA_DIR, file)


price_regex = re.compile(r"\b\d+\.{0,1}\d*\b")


def try_convert_price(val):
    # Convert mixed price data to float or nan
    if val == "None" or val is None:
        return np.nan
    try:
        return float(val)
    except ValueError:
        # Extract numeric portion, if possible
        res = price_regex.search(val)
        if res:
            return float(res.group(0))
        return np.nan


def try_datetime(val):
    try:
        # July, 19, 2022
        return datetime.datetime.strptime(val, "%B %d, %Y")
    except ValueError:
        try:
            # 2000-05-28
            return datetime.datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            # 09M 14, 2004
            return datetime.datetime.strptime(val, "%mM %d, %Y")
    except TypeError:
        return np.nan


def get_first_avail(val):
    # Extract first available date from details, if present
    try:
        return try_datetime(val["Date First Available"])
    except KeyError:
        return np.nan


def read_products_file(filename):
    if os.path.exists(get_data_filepath("products.par")):
        # Read preprocessed data from parquet file
        print("Reading pre-processed products data...")
        products = pd.read_parquet(get_data_filepath("products.par"))
    else:
        filepath = get_data_filepath(filename)
        if not os.path.exists(filepath):
            print(f"File not found!: {filepath}")
            return None

        # Read and preprocess main product data from json
        print("Pre-processing products data...")
        products = pd.read_json(filepath, lines=True)
        products.drop(
            columns=["images", "videos", "bought_together", "subtitle", "author"],
            inplace=True,
        )
        products["price"] = products["price"].apply(try_convert_price)
        products["first_available"] = products["details"].apply(get_first_avail)
        products.sort_values("price", ascending=False, inplace=True)
        print("Saving pre-processed data...")
        products.to_parquet(get_data_filepath("products.par"))

    print("Done")
    return products


def read_reviews_file(filename):
    if os.path.exists(get_data_filepath("reviews.par")):
        # Read preprocessed data from parquet file
        print("Reading pre-processed reviews data...")
        reviews = pd.read_parquet(get_data_filepath("reviews.par"))
    else:
        filepath = get_data_filepath(filename)
        if not os.path.exists(filepath):
            print(f"File not found!: {filepath}")
            return None

        # Read and preprocess main review data from json
        print("Pre-processing reviews data...")
        reviews = pd.read_json(filepath, lines=True)
        reviews.drop(columns=["images"], inplace=True)
        print("Saving pre-processed data...")
        reviews.to_parquet(get_data_filepath("reviews.par"))

    print("Done")
    return reviews


def summarize_column(
    df,
    colname,
    xlog=True,
    ylog=False,
    per="Product",
    axs=None,
    show=True,
    zipf=False,
    cumulative=True,
    stats=True,
    lmax=None,
):
    if axs is None:
        fig, ax = plt.subplots(1, 1)
        axs = fig.axes[0]
        fig.set_size_inches(6, 4)

    # Plot a histogram and some basic statistics info
    # about a numeric column in a dataframe
    fmean = df[colname].mean()
    fmedian = df[colname].median()
    fmax = df[colname].max()

    if xlog:
        if lmax is None:
            lmax = int(np.log10(2 * fmax) * 10) / 10
        counts, bins, bars = axs.hist(df[colname], np.logspace(0, lmax, 50))
    else:
        counts, bins, bars = axs.hist(df[colname])
    cmax = max(counts)

    if xlog:
        axs.set_xscale("log")
    if ylog:
        axs.set_yscale("log")

    axs.set_xlabel(f"{colname}" + (" (log)" if xlog else ""))
    axs.set_ylabel("Occurrence Count" + (" (log)" if ylog else ""))

    if stats:
        axs.vlines([fmean, fmedian, fmax], 0, cmax, "k", "dashed")

        idx = np.argmin(np.abs(bins - fmean))
        axs.text(
            max(1, fmean) * 1.1,
            counts[idx] + 0.01 * cmax,
            f"Mean:\n{fmean:,.0f}",
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
        )

        idx = np.argmin(np.abs(bins - fmedian))
        axs.text(
            max(1, fmedian) * 1.1,
            counts[idx] + 0.01 * cmax,
            f"Median:\n{fmedian:,.0f}",
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
        )

        idx = np.argmin(np.abs(bins - fmax))
        axs.text(
            fmax * 0.9,
            counts[idx] + 0.01 * cmax,
            f"Max:\n{fmax:,.0f}",
            ha="right",
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
        )

    if zipf:
        axs.plot([1, fmax], [cmax, 1], "k:")

    tmp = axs.get_ylim()
    if ylog:
        axs.set_ylim(tmp[0], tmp[1] * 2)
    else:
        axs.set_ylim(tmp[0], tmp[1] * 1.1)

    axs.set_title(f'Histogram of "{colname}" per {per}')

    if cumulative:
        ax2 = axs.twinx()
        ax2.plot(
            bins, [0] + (np.cumsum(counts) / np.sum(counts)).tolist(), "tab:orange"
        )
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("Cumulative Proportion")

    if show:
        plt.tight_layout()
        plt.show()


def print_missing(df):
    maxwidth = max([len(c) for c in df.columns])

    print("Data Missing")
    for col in df.columns:
        print(f"{col.ljust(maxwidth)}: {df[col].isna().sum()/len(df):.1%}")


def get_filament_products(df):
    # Match any known filament type
    filament_types = [
        "pla",
        "abs",
        "petg",
        "pc",
        "nylon",
        "pva",
        "hips",
        "asa",
        "tpu",
        "fpe",
        "pet",
        "pett",
        "pmma",
        "pom",
        "pp",
        "tpc",
        "tpe",
    ]
    regexes = f"(?:{'|'.join(filament_types)})"

    # Find all products with "filament" in their title
    filament_products = df[df["title"].str.lower().str.contains("filament")]

    # Filter to known filament types
    filament_products = filament_products[
        filament_products["title"].str.lower().str.contains(regexes, regex=True)
    ]

    # with pd.option_context('max_colwidth', 500):
    #     print(filament_products['title'])

    return filament_products
