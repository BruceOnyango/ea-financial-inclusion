"""Dash application: masthead, KPI cards, performance charts, and a live predictor.

View only. Every number comes from results.json (via app.data); every figure comes
from app.components. Callback bodies are thin module-level functions so they are
directly testable without a running server.
"""

from __future__ import annotations

from dash import ALL, Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from app import components, data, theme
from fia import config

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700&display=swap"
)

METRICS = ["roc_auc", "recall", "f1", "balanced_accuracy", "precision", "accuracy"]


# --------------------------------------------------------------------------- #
# small formatting helpers
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --------------------------------------------------------------------------- #
# layout builders (pure — take the results dict)
# --------------------------------------------------------------------------- #
def kpi_cards(results: dict) -> list[html.Div]:
    meta = results["meta"]
    tm = results["test_metrics"][meta["winner"]]
    rank = round(tm["roc_auc"] * 100)
    catch = round(tm["recall"] * 10)
    cards = [
        (
            "Adults with a bank account",
            _pct(meta["positive_rate"]),
            "So roughly 86 in every 100 adults sit outside the banking system.",
        ),
        (
            "People in this study",
            f"{meta['n_rows']:,}",
            f"Real survey respondents across {len(meta['countries'])} countries: "
            "Kenya, Rwanda, Tanzania and Uganda.",
        ),
        (
            "How well it tells them apart",
            f"{tm['roc_auc']:.2f}",
            f"Given one banked and one unbanked person, it ranks them correctly "
            f"about {rank} times in 100.",
        ),
        (
            "Banked adults it correctly finds",
            _pct(tm["recall"]),
            f"Of people who really do have an account, it catches about {catch} in 10.",
        ),
    ]
    return [
        html.Div(
            className="kpi",
            children=[
                html.Span(label, className="kpi-label"),
                html.Span(value, className="kpi-value"),
                html.Span(sub, className="kpi-sub"),
            ],
        )
        for label, value, sub in cards
    ]


def _form_field(col: str, opt: dict) -> html.Div:
    label = col.replace("_", " ").replace(" of respondent", "").capitalize()
    if opt["type"] == "numeric":
        control = dcc.Input(
            id={"type": "feat", "name": col},
            type="number",
            value=opt["default"],
            min=opt["min"],
            max=opt["max"],
            className="field-input",
        )
    else:
        control = dcc.Dropdown(
            id={"type": "feat", "name": col},
            options=[{"label": c, "value": c} for c in opt["choices"]],
            value=opt["choices"][0],
            clearable=False,
            className="field-drop",
        )
    return html.Div(
        className="field",
        children=[html.Label(label, className="field-label"), control],
    )


def dual_note(plain: str, technical: str) -> html.Div:
    """A two-line explainer: one plain-language, one technical."""
    return html.Div(
        className="note",
        children=[
            html.Div(
                className="note-row",
                children=[
                    html.Span("In plain terms", className="note-tag plain"),
                    html.Span(plain, className="note-text"),
                ],
            ),
            html.Div(
                className="note-row",
                children=[
                    html.Span("Technically", className="note-tag tech"),
                    html.Span(technical, className="note-text"),
                ],
            ),
        ],
    )


def build_layout(results: dict) -> html.Div:
    meta = results["meta"]
    model_names = config.ordered_models(results["confusion_matrices"].keys())
    drivers = list(results["eda_banked_rate"])

    return html.Div(
        className="page",
        children=[
            html.Header(
                className="masthead",
                children=[
                    html.Div(
                        className="masthead-inner",
                        children=[
                            html.Div(
                                [
                                    html.P(
                                        "FINANCIAL INCLUSION IN EAST AFRICA",
                                        className="eyebrow",
                                    ),
                                    html.H1(
                                        "Most adults here have no bank account. "
                                        "This tool helps find who, and why."
                                    ),
                                    html.P(
                                        "Across Kenya, Rwanda, Tanzania and Uganda, only about "
                                        "14 in every 100 adults hold a bank account. This "
                                        "dashboard uses survey data to show where exclusion is "
                                        "widest and to screen any individual in seconds.",
                                        className="masthead-sub",
                                    ),
                                ]
                            ),
                            html.Div(
                                className="masthead-tag",
                                children=[
                                    html.Span("PREDICTION MODEL"),
                                    html.Strong(meta["winner"]),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Main(
                className="container",
                children=[
                    html.Section(className="kpi-row", children=kpi_cards(results)),
                    # ---- 1. where the gaps are ----
                    html.Section(
                        className="block",
                        children=[
                            html.H2("Where the gaps are widest"),
                            html.P(
                                "The share of people with a bank account, split by the factors "
                                "that matter most. The dashed line is the regional average of "
                                "14%. Bars far below it mark the groups most left behind.",
                                className="lead",
                            ),
                            html.Div(
                                className="control-row",
                                children=[
                                    html.Label(
                                        "Break it down by:", className="control-label"
                                    ),
                                    dcc.Dropdown(
                                        id="eda-select",
                                        clearable=False,
                                        className="control-drop",
                                        value=drivers[0],
                                        options=[
                                            {"label": d.replace("_", " "), "value": d}
                                            for d in drivers
                                        ],
                                    ),
                                ],
                            ),
                            dcc.Graph(id="eda-graph", config={"displayModeBar": False}),
                            dual_note(
                                "Groups sitting well below the line, such as people with no "
                                "phone, little schooling, or who farm for a living, are where "
                                "inclusion efforts have the most ground to make up.",
                                "Each bar is the mean of the binary target within that category, "
                                "computed across all 23,524 records.",
                            ),
                        ],
                    ),
                    # ---- 2. screen an individual ----
                    html.Section(
                        className="block",
                        children=[
                            html.H2("Screen an individual"),
                            html.P(
                                "Enter someone's profile to estimate their chance of having a "
                                "bank account. A low chance means they are likely excluded, "
                                "and a good person to reach with an inclusion programme.",
                                className="lead",
                            ),
                            html.Div(
                                className="predictor",
                                children=[
                                    html.Div(
                                        className="form-grid",
                                        children=[
                                            _form_field(c, o)
                                            for c, o in results["form_options"].items()
                                        ],
                                    ),
                                    html.Button(
                                        "Estimate",
                                        id="predict-btn",
                                        n_clicks=0,
                                        className="btn",
                                    ),
                                    html.Div(
                                        id="prediction-result", className="result-empty"
                                    ),
                                ],
                            ),
                            dual_note(
                                "The result is a probability, not a certainty. It points you "
                                "toward who is most likely to need access, so outreach can be "
                                "focused rather than guessed.",
                                f"The model outputs P(account). A person is flagged banked when "
                                f"that probability is at or above the tuned threshold of "
                                f"{meta['winner_threshold']:.2f}.",
                            ),
                        ],
                    ),
                    # ---- 3. how reliable is this ----
                    html.Section(
                        className="block",
                        children=[
                            html.H2("How reliable is this?"),
                            html.P(
                                "Before trusting any prediction, it helps to see how the model "
                                "was tested. Five methods were compared and the strongest was "
                                "kept, judged on data it had never seen during training.",
                                className="lead",
                            ),
                            html.Div(
                                className="control-row",
                                children=[
                                    html.Label(
                                        "Compare the five models on:",
                                        className="control-label",
                                    ),
                                    dcc.Dropdown(
                                        id="metric-select",
                                        clearable=False,
                                        className="control-drop",
                                        value="roc_auc",
                                        options=[
                                            {
                                                "label": config.METRIC_LABELS[m],
                                                "value": m,
                                            }
                                            for m in METRICS
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="grid-2",
                                children=[
                                    dcc.Graph(
                                        id="metric-graph",
                                        config={"displayModeBar": False},
                                    ),
                                    dcc.Graph(
                                        id="roc-graph",
                                        config={"displayModeBar": False},
                                        figure=components.roc_overlay(
                                            results["roc_curves"]
                                        ),
                                    ),
                                ],
                            ),
                            dual_note(
                                "All five models get most people right, so a single accuracy "
                                "number hides the real difference. What separates them is how "
                                "well they spot the smaller group who are banked.",
                                "Accuracy is uninformative under a 14% positive rate. Models are "
                                "ranked on ROC-AUC (threshold-free ranking skill); the ROC curve "
                                "shows the trade-off between true and false positives.",
                            ),
                            html.Div(
                                className="control-row",
                                children=[
                                    html.Label(
                                        "See the detail for:", className="control-label"
                                    ),
                                    dcc.Dropdown(
                                        id="cm-select",
                                        clearable=False,
                                        className="control-drop",
                                        value=meta["winner"],
                                        options=[
                                            {"label": m, "value": m}
                                            for m in model_names
                                        ],
                                    ),
                                ],
                            ),
                            dcc.Graph(id="cm-graph", config={"displayModeBar": False}),
                            dual_note(
                                "This grid shows hits and misses: how many banked and unbanked "
                                "people the model got right, and where it slipped.",
                                "Confusion matrix on the held-out 30% test set at the tuned "
                                "threshold. The lower-left cell is missed banked adults (false "
                                "negatives), the cost we most want to keep low.",
                            ),
                        ],
                    ),
                    html.Footer(
                        className="foot",
                        children=[
                            html.P(
                                f"Data: FinScope and FinAccess surveys, 2016 to 2018. "
                                f"{meta['n_rows']:,} respondents. Model selected on ROC-AUC "
                                f"using {meta['cv_folds']}-fold cross-validation."
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# callback bodies (module-level, testable in isolation)
# --------------------------------------------------------------------------- #
def cb_metric(metric: str):
    return components.model_comparison_bar(data.get_results()["test_metrics"], metric)


def cb_confusion(model_name: str):
    results = data.get_results()
    return components.confusion_matrix_heatmap(
        results["confusion_matrices"][model_name], model_name
    )


def cb_eda(driver: str):
    rates = data.get_results()["eda_banked_rate"][driver]
    return components.banked_rate_bar(rates, driver.replace("_", " "))


def render_prediction(result: dict) -> html.Div:
    banked = result["label"] == 1
    cls = "result banked" if banked else "result not-banked"
    verdict = "Likely banked" if banked else "Likely unbanked"
    return html.Div(
        className=cls,
        children=[
            html.Span(verdict, className="result-verdict"),
            html.Span(
                f"P(has account) = {result['probability']:.1%}", className="result-prob"
            ),
            html.Span(
                f"decision threshold {result['threshold']:.1%}", className="result-thr"
            ),
        ],
    )


def cb_predict(n_clicks, values, ids):
    if not n_clicks:
        raise PreventUpdate
    record = {i["name"]: v for i, v in zip(ids, values)}
    return render_prediction(data.predict_record(record)), "result-shown"


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def register_callbacks(app: Dash) -> Dash:
    app.callback(Output("metric-graph", "figure"), Input("metric-select", "value"))(
        cb_metric
    )
    app.callback(Output("cm-graph", "figure"), Input("cm-select", "value"))(
        cb_confusion
    )
    app.callback(Output("eda-graph", "figure"), Input("eda-select", "value"))(cb_eda)
    app.callback(
        [
            Output("prediction-result", "children"),
            Output("prediction-result", "className"),
        ],
        Input("predict-btn", "n_clicks"),
        State({"type": "feat", "name": ALL}, "value"),
        State({"type": "feat", "name": ALL}, "id"),
    )(cb_predict)
    return app


def create_app() -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[FONTS],
        title="Financial Inclusion · East Africa",
        url_base_pathname="/financial-inclusion/",
    )
    app.layout = build_layout(data.get_results())
    register_callbacks(app)
    return app


if __name__ == "__main__":
    create_app().run(debug=True)
