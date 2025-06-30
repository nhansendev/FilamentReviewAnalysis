import pandas as pd
import numpy as np
import ipywidgets as widgets
from tqdm import tqdm
from constants import RANDOM_SEED, RNG
import matplotlib.pyplot as plt
from functools import partial

from umap import UMAP
from textwrap import wrap
from bertopic import BERTopic
from sklearn.cluster import AffinityPropagation, MiniBatchKMeans
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer


def print_cat_statistics(df, col="relevant"):
    tmp = df[col].value_counts(dropna=False)
    for i in tmp.index:
        print(f"{i}: {tmp[i]} ({tmp[i]/len(df):.1%})")


def iterate_self_learning(df, model, clf, thr=0.95):
    # Target the non-labeled subset of data
    subset = df["relevant"].isna()

    # Separate the already labeled data for training
    labeled_titles = df[~subset]
    labels = labeled_titles["relevant"].astype(int).values
    labeled_titles = labeled_titles["title"].values

    # Embed the non-labeled and labeled data
    all_embeddings = model.encode(
        df.loc[subset, "title"].values, convert_to_tensor=False
    )
    labeled_embeddings = model.encode(labeled_titles, convert_to_tensor=False)

    # Fit the regression model to the labeled data
    clf.fit(labeled_embeddings, labels)

    # Update the non-labled data with predicted labels
    # where confidence is over a set threshold

    # Get predictions and confidence per prediction
    probs = clf.predict_proba(all_embeddings)
    confidences = np.max(probs, axis=1)
    confident_preds = confidences > thr
    preds = clf.predict(all_embeddings).astype(bool)

    # A bit tricky: we need the subset of non-labeled data
    # and the subset of that subset with high confidence
    targets = df["relevant"].isna()
    targets[targets] *= confident_preds

    # Assign high-confidence predicted labels to matching non-labled data
    df.loc[targets, "relevant"] = preds[confident_preds]

    # Show assignment statistics
    print_cat_statistics(df)


def run_self_learning(df, model, clf, iters=3, thr=0.95):
    for _ in tqdm(range(iters)):
        iterate_self_learning(df, model, clf, thr=thr)


def get_topic_top_tfidf(series, tokenizer, doc_count_map, N=10, ignore_single_doc=True):
    subset = pd.DataFrame(series.apply(tokenizer).explode().value_counts())
    subset.index.name = None
    subset["tfidf"] = subset["count"] / doc_count_map.loc[subset.index, "count"]
    subset.sort_values("tfidf", ascending=False, inplace=True)
    if ignore_single_doc:
        # Ignore terms that ONLY appear in this group doc_freq = 1, so tfidf == count
        return (
            subset[subset["tfidf"].astype(int) != subset["count"]]
            .head(N)
            .index.tolist()
        )
    else:
        # Return top N of everything
        return subset.head(N).index.tolist()


class VarHolder:
    def __init__(self, values, reverse=False):
        self.values = values
        self.reverse = reverse
        self.idx = len(values) - 1 if reverse else 0

    @property
    def val(self):
        return self.values[self.idx]

    def iter(self):
        if self.reverse:
            self.dec()
        else:
            self.inc()

    def rev(self):
        if self.reverse:
            self.inc()
        else:
            self.dec()

    def inc(self):
        self.idx += 1
        if self.idx >= len(self.values):
            self.idx = 0

    def dec(self):
        self.idx -= 1
        if self.idx < 0:
            self.idx = len(self.values) - 1


def manual_inspect(
    df,
    tokenizer,
    doc_count_map,
    clusterer=None,
    clustering_fn=None,
    centers=None,
    order=None,
    common_term_thr=0.95,
    target_cluster_size=10,
    max_entries=50,
):
    uniques = np.unique(df["topic"])
    # Ignore nan and -100 (placeholder value)
    uniques = uniques[np.logical_and(~np.isnan(uniques), uniques != -100)]
    uniques = uniques.astype(int)
    if clusterer:
        idxs = VarHolder(get_cluster_eval_order(clusterer.cluster_centers_))
    elif centers is not None:
        idxs = VarHolder(get_cluster_eval_order(centers))
    elif order == "rand":
        seq = list(range(min(uniques), max(uniques) + 1))
        RNG.shuffle(seq)
        idxs = VarHolder(seq)
    elif order is not None:
        idxs = VarHolder(order)
    else:
        idxs = VarHolder(list(range(min(uniques), max(uniques) + 1)), reverse=True)

    # out = widgets.Output()
    out = widgets.HTML()
    msg = widgets.Text(layout=widgets.Layout(width="500px"))

    def f(*args, iterate=True, reverse=False):
        if iterate:
            if reverse:
                idxs.rev()
            else:
                idxs.iter()

        out.value = ""

        idx = idxs.val
        if np.sum(df["topic"] == idx) < 1:
            # Empty
            out.value = f"#{idx}: No entries..."
        else:
            is_marked = df.loc[df["topic"] == idx, "relevant"].isna().sum() == 0

            subset = df[df["topic"] == idx]["title"]
            out.value = (
                r'<p style="font-size:16px; ">'
                + f"#{idx}, Marked: {is_marked}, Items: {len(subset)}<br><br>"
                + print_topic_summarized_formatted(
                    subset,
                    tokenizer,
                    clustering_fn=clustering_fn,
                    common_thr=common_term_thr,
                    target_cluster_size=target_cluster_size,
                    max_entries=max_entries,
                )
                + r"</p>"
            )

        tmp = df["relevant"].value_counts(dropna=False)
        line = "   |   ".join(
            [f"{i}: {tmp[i]} ({tmp[i]/len(df):.1%})" for i in tmp.index]
        )
        msg.value = line

    def mark_relevant(*args):
        df.loc[df["topic"] == idxs.val, "relevant"] = True
        f()

    def mark_irrelevant(*args):
        df.loc[df["topic"] == idxs.val, "relevant"] = False
        f()

    def clear_mark(*args):
        df.loc[df["topic"] == idxs.val, "relevant"] = np.nan
        f()

    def _reverse(*args):
        f(reverse=True)

    next_button = widgets.Button(description="Skip")
    next_button.on_click(f)

    back_button = widgets.Button(description="Back")
    back_button.on_click(_reverse)

    irrel_button = widgets.Button(description="Irrelevant")
    irrel_button.on_click(mark_irrelevant)

    rel_button = widgets.Button(description="Relevant")
    rel_button.on_click(mark_relevant)

    clear_button = widgets.Button(description="Clear")
    clear_button.on_click(clear_mark)

    button_box = widgets.HBox(
        [next_button, back_button, irrel_button, rel_button, clear_button, msg]
    )
    main_layout = widgets.VBox([button_box, out])

    f(iterate=False)

    return main_layout


def manual_individual_review(df):
    def _process_target(target, state, *args):
        df.loc[target, "relevant"] = state

    # Print the remaining items
    target = df["relevant"].isna()
    idxs = df.index[target].values
    titles = df.loc[target, "title"].values

    header = widgets.Text()
    header.value = f"{len(titles)} Entries Remaining:"
    entries = [
        WidgetEntry(f"#{i}: {t}", idxs[i], _process_target)
        for i, t in enumerate(titles)
    ]
    main_layout = widgets.VBox([header] + [e.layout for e in entries])

    for e in entries:
        e.show()

    return main_layout


class WidgetEntry:
    def __init__(self, text="", target=None, clickfunc=None):
        self.target = target
        self.clickfunc = clickfunc

        self.text = widgets.Text()
        self.text.layout = widgets.Layout(width="2000px")
        self.text.value = text

        self.irrel_button = widgets.Button(description="Irrelevant")
        self.irrel_button.on_click(self._irr_click)

        self.rel_button = widgets.Button(description="Relevant")
        self.rel_button.on_click(self._rel_click)

        self.button_layout = widgets.HBox([self.rel_button, self.irrel_button])
        self.layout = widgets.HBox([self.button_layout, self.text])

        self.hide()

    def _rel_click(self, _):
        if self.clickfunc:
            self.clickfunc(self.target, True)
        self.text.style.background = "lightgreen"

    def _irr_click(self, _):
        if self.clickfunc:
            self.clickfunc(self.target, False)
        self.text.style.background = "red"

    def reset(self):
        self.text.clear_output()
        self.rel_button._click_handlers = widgets.CallbackDispatcher()
        self.irrel_button._click_handlers = widgets.CallbackDispatcher()

    def show(self):
        self.text.layout.display = ""
        self.rel_button.layout.display = ""
        self.irrel_button.layout.display = ""

    def hide(self):
        self.text.layout.display = "none"
        self.rel_button.layout.display = "none"
        self.irrel_button.layout.display = "none"


def manual_sep_inspect(
    df,
    tokenizer,
    clusterer=None,
    clustering_fn=None,
    centers=None,
    common_term_thr=0.95,
    max_entries=30,
):
    uniques = np.unique(df["topic"])
    # Ignore nan and -100 (placeholder value)
    uniques = uniques[np.logical_and(~np.isnan(uniques), uniques != -100)]
    if clusterer:
        idxs = VarHolder(get_cluster_eval_order(clusterer.cluster_centers_))
    elif centers is not None:
        idxs = VarHolder(get_cluster_eval_order(centers))
    else:
        idxs = VarHolder(list(range(min(uniques), max(uniques) + 1)))

    header = widgets.Output()
    msg = widgets.Text(layout=widgets.Layout(width="500px"))

    entries = [WidgetEntry() for _ in range(max_entries)]

    def f(*args):
        header.clear_output()
        for e in entries:
            e.hide()

        idx = idxs.val
        if np.sum(df["topic"] == idx) < 1:
            # Empty
            with header:
                print(f"#{idx}: No entries...")
        else:
            is_marked = df.loc[df["topic"] == idx, "relevant"].isna().sum() == 0

            with header:
                subset = df[df["topic"] == idx]["title"]
                print(f"#{idx}, Marked: {is_marked}, Items: {len(subset)}")

                lines, masks = print_topic_summarized(
                    subset,
                    tokenizer,
                    clustering_fn=clustering_fn,
                    common_thr=common_term_thr,
                    tostr=True,
                )
                print()

            for i, line in enumerate(lines):
                entries[i].reset()
                target = df["topic"] == idx
                target[target] *= masks[i]

                # This seems to be the only way to make the dataframe
                # update with these button presses
                def _r(*args):
                    df.loc[target, "relevant"] = True

                def _ir(*args):
                    df.loc[target, "relevant"] = False

                entries[i].rel_button.on_click(_r)
                entries[i].irrel_button.on_click(_ir)
                entries[i].show()
                with entries[i].text:
                    print(line)

        tmp = df["relevant"].value_counts(dropna=False)
        line = "   |   ".join(
            [f"{i}: {tmp[i]} ({tmp[i]/len(df):.1%})" for i in tmp.index]
        )
        msg.value = line

        idxs.inc()

    def mark_relevant(*args):
        df.loc[df["topic"] == idxs.val, "relevant"] = True
        f()

    def mark_irrelevant(*args):
        df.loc[df["topic"] == idxs.val, "relevant"] = False
        f()

    next_button = widgets.Button(description="Skip")
    next_button.on_click(f)

    irrel_button = widgets.Button(description="Irrelevant")
    irrel_button.on_click(mark_irrelevant)

    rel_button = widgets.Button(description="Relevant")
    rel_button.on_click(mark_relevant)

    button_box = widgets.HBox([next_button, irrel_button, rel_button, msg])
    main_layout = widgets.VBox([button_box, header] + [e.layout for e in entries])

    f()

    return main_layout


def gen_topics(
    df,
    min_topic_size=30,
    num_topics=None,
    cluster_model=None,
    col="title",
    return_model=False,
    semi_supervised=False,
    embeddings=None,
):

    # Only needed to set random seed of BERTopic
    # Uses the default values used internally by BERTopic
    umap = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        low_memory=False,
        random_state=RANDOM_SEED,
    )

    topic_model = BERTopic(
        language="english",
        verbose=True,
        min_topic_size=min_topic_size,
        nr_topics=num_topics,
        hdbscan_model=cluster_model,
        umap_model=umap,
    )

    unassigned = df["relevant"].isna()
    if semi_supervised:
        # Semi-supervised
        tmp = df["relevant"].values
        tmp[unassigned] = -1

        # Generate topics for the entries that are still nan
        topics, probs = topic_model.fit_transform(df[col], y=tmp, embeddings=embeddings)
    else:
        # Unsupervised
        # Generate topics for the entries that are still nan
        topics, probs = topic_model.fit_transform(df[col], embeddings=embeddings)

    if isinstance(cluster_model, BernoulliNB_Clusterer):
        df["topic"] = topics
    else:
        # Reset topics that were already generated to a common placeholder (-100)
        df.loc[~unassigned, "topic"] = -100
        # Fill all unassigned topics with the new values
        df.loc[unassigned, "topic"] = np.array(topics)[unassigned]

    if return_model:
        return topic_model.topic_embeddings_, topic_model
    else:
        return topic_model.topic_embeddings_

    # if hasattr(topic_model.hdbscan_model, "cluster_centers_"):
    #     return topic_model.hdbscan_model.cluster_centers_

    # if hasattr(topic_model.hdbscan_model, "weighted_cluster_centroid"):
    #     return [
    #         topic_model.hdbscan_model.weighted_cluster_centroid(i)
    #         for i in np.unique(topics)
    #         if i != -1
    #     ]


def simple_tokenize(data):
    tokenizer = TfidfVectorizer().build_analyzer()
    if isinstance(data, pd.DataFrame):
        return data["title"].str.lower().apply(tokenizer)
    if isinstance(data, list):
        return [tokenizer(t.lower()) for t in data]


def vectorize_titles(df, verbose=False):
    # Encode titles to vectors for tfidf analysis
    vectorizer = TfidfVectorizer()
    # Tokenizer will be reused later for topic-by-topic tfidf
    tokenizer = vectorizer.build_analyzer()
    vecs = vectorizer.fit_transform(df["title"].str.lower().tolist())
    names = vectorizer.get_feature_names_out()

    if verbose:
        print(vecs.shape)
        print(names)

    # Count how many documents each term appears in (only count once per doc)
    doc_counts = np.asarray((vecs > 0).sum(0)).flatten()
    # Map term to document count
    doc_count_map = pd.DataFrame(doc_counts, index=names, columns=["count"])

    return tokenizer, doc_count_map


def get_cluster_eval_order(centers, ascending=False):
    n_clusters = centers.shape[0]
    # Calculate squared differences between cluster centers
    cluster_diff = [[] for _ in range(n_clusters)]
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            diff = np.sum((centers[i] - centers[j]) ** 2)
            cluster_diff[i].append(diff)
            cluster_diff[j].append(diff)

    # Provide the topic indexes in ascending or descending order
    # of average difference from all other topics based on embeddings.
    # Put another way: get the most unique, or common topics first per their embeddings
    if ascending:
        return np.argsort(np.mean(np.array(cluster_diff), 1)).tolist()
    else:
        return np.argsort(np.mean(np.array(cluster_diff), 1)).tolist()[::-1]


def print_topic_summarized(
    subset,
    tokenizer,
    clustering_fn=None,
    target_cluster_size=10,
    maxwidth=150,
    common_thr=0.95,
    verbose=False,
    tostr=False,
):
    if tokenizer is None:
        tokenizer = TfidfVectorizer().build_analyzer()

    sep_subset = subset.apply(tokenizer)

    uniques = sep_subset.explode().unique()
    # Simple vector encoding
    unique_map = {k: v for v, k in enumerate(uniques)}

    encoded = np.zeros((len(sep_subset), len(unique_map)))
    if verbose:
        print(encoded.shape)

    for i, entry in enumerate(sep_subset.values):
        for term in np.unique(entry):
            encoded[i][unique_map[term]] = 1

    if clustering_fn:
        try:
            db = clustering_fn.fit(encoded)
        except ValueError:
            db = AffinityPropagation(damping=0.99).fit(encoded)
    else:
        # db = AffinityPropagation().fit(encoded)
        clusters = min(len(sep_subset), 1 + int(len(sep_subset) / target_cluster_size))
        db = MiniBatchKMeans(clusters, random_state=RANDOM_SEED).fit(encoded)

    if verbose:
        print(np.unique(db.labels_, return_counts=True))

    uniques_labels = np.unique(db.labels_)
    print(f"Identified {len(uniques_labels)} Cluster(s):")

    out = []
    index_masks = []
    for idx in uniques_labels:
        tmp = encoded[db.labels_ == idx]
        index_masks.append(db.labels_ == idx)

        thr = tmp.sum(0) >= common_thr * tmp.shape[0]
        thr2 = np.logical_and(~thr, tmp.sum(0) >= 1)

        # Common
        comm = " ".join(uniques[thr])
        # Different
        diff = ", ".join(uniques[thr2].tolist())

        comm_lines = wrap(comm, maxwidth)
        diff_lines = wrap(diff, maxwidth)

        out.append("")
        if len(comm) < 1:
            if tostr:
                out[
                    -1
                ] += f"#{str(idx).rjust(3)}, Items: {len(tmp)}, : {diff_lines[0]}\n"
                for line in diff_lines[1:]:
                    out[-1] += "   >" + line + "\n"
            else:
                print(f"\n#{str(idx).rjust(3)}, Items: {len(tmp)}, : {diff_lines[0]}")
                for line in diff_lines[1:]:
                    print("   >", line)
        else:
            if tostr:
                out[
                    -1
                ] += f"#{str(idx).rjust(3)}, Items: {len(tmp)}, : {comm_lines[0]}\n"
                for line in comm_lines[1:]:
                    out[-1] += "   >" + line + "\n"

                for line in diff_lines:
                    out[-1] += "      -" + line + "\n"
            else:
                print(f"\n#{str(idx).rjust(3)}, Items: {len(tmp)}, : {comm_lines[0]}")
                for line in comm_lines[1:]:
                    print("   >", line)

                for line in diff_lines:
                    print("      -", line)

    return out, index_masks


def print_topic_summarized_formatted(
    subset,
    tokenizer,
    clustering_fn=None,
    target_cluster_size=10,
    maxwidth=150,
    common_thr=0.95,
    verbose=False,
    tostr=False,
    max_entries=50,
):
    if tokenizer is None:
        tokenizer = TfidfVectorizer().build_analyzer()

    sep_subset = subset.apply(tokenizer)

    uniques = sep_subset.explode().unique()
    # Simple vector encoding
    unique_map = {k: v for v, k in enumerate(uniques)}

    encoded = np.zeros((len(sep_subset), len(unique_map)))

    for i, entry in enumerate(sep_subset.values):
        for term in np.unique(entry):
            encoded[i][unique_map[term]] = 1

    if clustering_fn:
        try:
            db = clustering_fn.fit(encoded)
        except ValueError:
            db = AffinityPropagation(damping=0.99).fit(encoded)
    else:
        # db = AffinityPropagation().fit(encoded)
        clusters = min(len(sep_subset), 1 + int(len(sep_subset) / target_cluster_size))
        db = MiniBatchKMeans(clusters, random_state=RANDOM_SEED).fit(encoded)

    if verbose:
        print(np.unique(db.labels_, return_counts=True))

    uniques_labels = np.unique(db.labels_)

    def _repl(terms):
        out = ""
        for term in terms:
            if term in comm:
                out += rf" <b>{term}</b>"
            else:
                out += f" {term}"
        return out

    out = [f"Identified {len(uniques_labels)} Cluster(s):"]
    for idx in uniques_labels[:max_entries]:
        out.append("")
        tmp = encoded[db.labels_ == idx]
        thr = tmp.sum(0) >= common_thr * tmp.shape[0]

        # Common
        comm = uniques[thr]
        titles = (
            sep_subset[db.labels_ == idx]
            .sort_values(key=lambda x: x.str.len())
            .apply(_repl)
        )

        out[-1] += f"#{str(idx).rjust(3)}, Items: {len(tmp)}, : {comm}<br>"
        for title in titles:
            out[-1] += "      -" + title + "<br>"
    return "<br>".join(out)


def assign_encoded_topics(df, encoded, clustering_fn):
    na_idxs = df["relevant"].isna()

    if clustering_fn:
        try:
            db = clustering_fn.fit(encoded)
        except ValueError:
            db = AffinityPropagation(damping=0.9).fit(encoded)
    else:
        db = AffinityPropagation().fit(encoded)

    df.loc[~na_idxs, "topic"] = -100
    # Fill all unassigned topics with the new values
    df.loc[na_idxs, "topic"] = db.labels_


def assign_topics_simple(df, tokenizer, clustering_fn):
    na_idxs = df["relevant"].isna()
    subset = df.loc[na_idxs, "title"]

    sep_subset = subset.apply(tokenizer)

    uniques = sep_subset.explode().unique()
    # Simple vector encoding
    unique_map = {k: v for v, k in enumerate(uniques)}

    encoded = np.zeros((len(sep_subset), len(unique_map)))

    for i, entry in enumerate(sep_subset.values):
        for term in np.unique(entry):
            encoded[i][unique_map[term]] = 1

    if clustering_fn:
        try:
            db = clustering_fn.fit(encoded)
        except ValueError:
            db = AffinityPropagation(damping=0.9).fit(encoded)
    else:
        db = AffinityPropagation().fit(encoded)

    df.loc[~na_idxs, "topic"] = -100
    # Fill all unassigned topics with the new values
    df.loc[na_idxs, "topic"] = db.labels_


def plot_generated_topics(df, centroids, ylog=False, size=(12, 4)):
    values, counts = np.unique(df["topic"], return_counts=True)
    valid = values > -2

    # print(len(values), len(centroids))

    # Order topics by centroid dissimilarity
    # Ascending means most similar first
    order = get_cluster_eval_order(np.array(centroids)[valid], ascending=False)

    ordered_topics = [int(values[valid][i]) for i in order]

    plt.bar(values[valid], counts[valid])
    plt.xlabel("Topic ID")
    plt.ylabel("Count")
    if ylog:
        plt.yscale("log")
    plt.title("Generated Topics")

    plt.gcf().set_size_inches(size)

    ax2 = plt.gca().twinx()
    ax2.bar(values[valid], np.argsort(ordered_topics), color="r", alpha=0.4)
    ax2.set_ylabel("Topic Avg Similarity Rank")
    plt.show()

    return ordered_topics


def simple_vector_encode(df):
    tokenizer = TfidfVectorizer().build_analyzer()
    sep_subset = df["title"].apply(tokenizer)

    uniques = sep_subset.explode().unique()
    # Simple vector encoding
    unique_map = {k: v for v, k in enumerate(uniques)}

    encoded = np.zeros((len(sep_subset), len(unique_map)))

    for i, entry in enumerate(sep_subset.values):
        for term in np.unique(entry):
            encoded[i][unique_map[term]] = 1

    return encoded, uniques


class BernoulliNB_Clusterer:
    def __init__(
        self,
        fit_mask,
        known_labels,
        cluster_size=50,
        clusterer=None,
        verbose=True,
    ):
        # Semi-supervised clustering of two classes

        self.fit_mask = fit_mask
        self.known_labels = known_labels
        self.embeddings = None
        self.nb = BernoulliNB()
        self.cluster_size = cluster_size
        self.random_seed = RANDOM_SEED
        if clusterer:
            self.clusterer = clusterer
        else:
            self.clusterer = MiniBatchKMeans
        self.verbose = verbose

    def fit(self, X):
        # Train Bernoulli Naive Bayes model
        self.preds = None
        self.embeddings = X
        self.nb.fit(X[self.fit_mask], self.known_labels.astype(int))
        return self.predict(X)

    def update_fit(self, fit_mask, known_labels):
        assert (
            self.embeddings is not None
        ), "Must perform fit first, or provide embeddings"
        self.fit_mask = fit_mask
        self.known_labels = known_labels
        self.nb.fit(self.embeddings[fit_mask], known_labels.astype(int))
        return self.predict(self.embeddings)

    def predict(self, X):
        # Predict new class labels with fitted BernoulliNB model
        preds = self.nb.predict(X) > 0.5
        # Unlabeled subset
        subsetA = preds * ~self.fit_mask
        subsetB = ~preds * ~self.fit_mask

        # Use clustering algorithm to split classes into sub-classes
        n_clustersA = max(2, int(np.ceil(sum(subsetA) / self.cluster_size)))
        A_labels = self.clusterer(
            n_clustersA, random_state=self.random_seed
        ).fit_predict(X[subsetA])

        n_clustersB = max(2, int(np.ceil(sum(subsetB) / self.cluster_size)))
        B_labels = (
            self.clusterer(n_clustersB, random_state=self.random_seed).fit_predict(
                X[subsetB]
            )
            + max(A_labels)
            + 1
        )

        if self.verbose:
            print("Cluster Sizes:")
            print(f"True: sub-clusters: {n_clustersA}, items: {len(A_labels)}")
            if self.verbose > 1:
                values, counts = np.unique(A_labels, return_counts=True)
                for v, c in zip(values, counts):
                    print(f"{v}: {c}")
            print(f"False: sub-clusters: {n_clustersB}, items: {len(B_labels)}")
            if self.verbose > 1:
                values, counts = np.unique(B_labels, return_counts=True)
                for v, c in zip(values, counts):
                    print(f"{v}: {c}")

        out = np.ones(len(X)) * -100
        out[subsetA] = A_labels
        out[subsetB] = B_labels

        self.labels_ = out

        return out
