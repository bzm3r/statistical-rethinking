import os
import polars as pl
import numpy as np
# import numpy as np
import pymc as pm
# import arviz as az
import graphviz as gv
from pathlib import Path
from typing import List, Optional

HERE = Path(".")


def load_data(dataset, delimiter=";"):
    fname = f"{dataset}.csv"
    data_path = HERE / "data"
    data_file = data_path / fname

    return pl.read_csv(data_file, separator=delimiter)


def crosstab(xs: pl.Series, ys: pl.Series) -> pl.DataFrame:
    """Cross tabulation of two series."""
    return (
        pl.DataFrame(xs).join(pl.DataFrame(ys), how="cross").with_columns(count=pl.lit(1, dtype=pl.Int64()))
        .group_by("x", "y")
        .agg(pl.col("count").sum())
        )


def center_across_mean(xs: pl.Series) -> pl.Series:
    """Center given series across arithmetic mean."""
    return xs - xs.drop_nulls().mean()


def normalize_centered_by_std(xs: pl.Series) -> pl.Series:
    """"Standardize" given series: i.e. first center_across_mean then normalize
    by standard deviation of centered values."""
    centered = center_across_mean(xs)
    return centered / centered.drop_nulls().std()


def convert_to_enum(xs: pl.Series) -> pl.Series:
    return pl.Series(xs, dtype=pl.Enum(xs.drop_nulls().unique().sort()))

def logit(xs: pl.Series) -> pl.Series:
    return (xs / (1 - xs)).log()

def invlogit(x: pl.Series) -> pl.Series:
    return 1 / (1 + (-1 * x).exp())

# TODO: use something sane instead of networkx
# def draw_causal_graph(
#     edge_list, node_props=None, edge_props=None, graph_direction="UD"
# ):
#     """Utility to draw a causal (directed) graph"""
#     g = gv.Digraph(graph_attr={"rankdir": graph_direction})

#     edge_props = {} if edge_props is None else edge_props
#     for e in edge_list:
#         props = edge_props[e] if e in edge_props else {}
#         g.edge(e[0], e[1], **props)

#     if node_props is not None:
#         for name, props in node_props.items():
#             g.node(name=name, **props)

#     return g

# TODO: use something sane instead of networkx
# def plot_graph(graph, **graph_kwargs):
#     """Draw a network graph.

#     graph: Union[networkx.DiGraph, np.ndarray]
#         if ndarray, assume `graph` is an adjacency matrix defining
#         a directed graph.

#     """
#     # convert to networkx.DiGraph, if needed
#     G = (
#         nx.from_numpy_array(graph, create_using=nx.DiGraph)
#         if isinstance(graph, np.ndarray)
#         else graph
#     )

#     # Set default styling
#     np.random.seed(123)  # for consistent spring-layout
#     if "layout" in graph_kwargs:
#         graph_kwargs["pos"] = graph_kwargs["layout"](G)

#     default_graph_kwargs = {
#         "node_color": "C0",
#         "node_size": 500,
#         "arrowsize": 30,
#         "width": 3,
#         "alpha": 0.7,
#         "connectionstyle": "arc3,rad=0.1",
#         "pos": nx.kamada_kawai_layout(G),
#     }
#     for k, v in default_graph_kwargs.items():
#         if k not in graph_kwargs:
#             graph_kwargs[k] = v

#     nx.draw(G, **graph_kwargs)
#     # return the node layout for consistent graphing
#     return graph_kwargs["pos"]

