import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

MODEL_PATH = Path(__file__).parent / "final_phishing_model.pkl"

CHARACTER_MAP = {
    "dot": ".",
    "hyphen": "-",
    "underline": "_",
    "slash": "/",
    "questionmark": "?",
    "equal": "=",
    "at": "@",
    "and": "&",
    "exclamation": "!",
    "space": " ",
    "tilde": "~",
    "comma": ",",
    "plus": "+",
    "asterisk": "*",
    "hashtag": "#",
    "dollar": "$",
    "percent": "%",
}

PLACEHOLDER_FEATURES = [
    "time_response",
    "domain_spf",
    "asn_ip",
    "time_domain_activation",
    "time_domain_expiration",
    "qty_ip_resolved",
    "qty_nameservers",
    "qty_mx_servers",
    "ttl_hostname",
    "tls_ssl_certificate",
    "qty_redirects",
    "url_google_index",
    "domain_google_index",
]

SHORTENER_PATTERNS = [
    "bit.ly",
    "tinyurl",
    "goo.gl",
    "ow.ly",
    "t.co",
    "is.gd",
    "buff.ly",
    "adf.ly",
]


@st.cache_data
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    if hasattr(pipeline, "named_steps"):
        scaler = pipeline.named_steps.get("standardscaler")
        estimator = pipeline.named_steps[pipeline.steps[-1][0]]
        feature_columns = list(
            getattr(scaler, "feature_names_in_", getattr(pipeline, "feature_names_in_", []))
        )
    else:
        scaler = None
        estimator = pipeline
        feature_columns = list(getattr(pipeline, "feature_names_in_", []))

    if not feature_columns:
        raise ValueError("Could not determine feature columns from the loaded model.")

    return pipeline, scaler, estimator, feature_columns


def count_char(text: str, char: str) -> int:
    return text.count(char) if isinstance(text, str) else 0


def is_ip_address(domain: str) -> bool:
    return bool(re.fullmatch(r"[0-9.]+", domain))


def extract_url_features(url: str, columns: list[str]) -> pd.DataFrame:
    url = url.strip()
    parsed = urlparse(url)
    domain = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""
    file_name = path.split("/")[-1] if path else ""
    directory = "/".join(path.split("/")[1:-1]) if "/" in path else ""

    features = {}
    field_values = {
        "url": url,
        "domain": domain,
        "directory": directory,
        "file": file_name,
        "params": query,
    }

    for field, value in field_values.items():
        for token, symbol in CHARACTER_MAP.items():
            features[f"qty_{token}_{field}"] = count_char(value, symbol)

    features.update(
        {
            "qty_tld_url": len([part for part in domain.split(".") if part]) - 1 if domain else 0,
            "length_url": len(url),
            "qty_vowels_domain": sum(domain.lower().count(v) for v in "aeiou"),
            "domain_length": len(domain),
            "domain_in_ip": int(is_ip_address(domain)),
            "server_client_domain": 0,
            "qty_dot_directory": count_char(directory, "."),
            "qty_hyphen_directory": count_char(directory, "-"),
            "qty_underline_directory": count_char(directory, "_"),
            "qty_slash_directory": count_char(directory, "/"),
            "qty_questionmark_directory": count_char(directory, "?"),
            "qty_equal_directory": count_char(directory, "="),
            "qty_at_directory": count_char(directory, "@"),
            "qty_and_directory": count_char(directory, "&"),
            "qty_exclamation_directory": count_char(directory, "!"),
            "qty_space_directory": count_char(directory, " "),
            "qty_tilde_directory": count_char(directory, "~"),
            "qty_comma_directory": count_char(directory, ","),
            "qty_plus_directory": count_char(directory, "+"),
            "qty_asterisk_directory": count_char(directory, "*"),
            "qty_hashtag_directory": count_char(directory, "#"),
            "qty_dollar_directory": count_char(directory, "$"),
            "qty_percent_directory": count_char(directory, "%"),
            "directory_length": len(directory),
            "qty_dot_file": count_char(file_name, "."),
            "qty_hyphen_file": count_char(file_name, "-"),
            "qty_underline_file": count_char(file_name, "_"),
            "qty_slash_file": count_char(file_name, "/"),
            "qty_questionmark_file": count_char(file_name, "?"),
            "qty_equal_file": count_char(file_name, "="),
            "qty_at_file": count_char(file_name, "@"),
            "qty_and_file": count_char(file_name, "&"),
            "qty_exclamation_file": count_char(file_name, "!"),
            "qty_space_file": count_char(file_name, " "),
            "qty_tilde_file": count_char(file_name, "~"),
            "qty_comma_file": count_char(file_name, ","),
            "qty_plus_file": count_char(file_name, "+"),
            "qty_asterisk_file": count_char(file_name, "*"),
            "qty_hashtag_file": count_char(file_name, "#"),
            "qty_dollar_file": count_char(file_name, "$"),
            "qty_percent_file": count_char(file_name, "%"),
            "file_length": len(file_name),
            "qty_dot_params": count_char(query, "."),
            "qty_hyphen_params": count_char(query, "-"),
            "qty_underline_params": count_char(query, "_"),
            "qty_slash_params": count_char(query, "/"),
            "qty_questionmark_params": count_char(query, "?"),
            "qty_equal_params": count_char(query, "="),
            "qty_at_params": count_char(query, "@"),
            "qty_and_params": count_char(query, "&"),
            "qty_exclamation_params": count_char(query, "!"),
            "qty_space_params": count_char(query, " "),
            "qty_tilde_params": count_char(query, "~"),
            "qty_comma_params": count_char(query, ","),
            "qty_plus_params": count_char(query, "+"),
            "qty_asterisk_params": count_char(query, "*"),
            "qty_hashtag_params": count_char(query, "#"),
            "qty_dollar_params": count_char(query, "$"),
            "qty_percent_params": count_char(query, "%"),
            "params_length": len(query),
            "tld_present_params": int("." in query),
            "qty_params": len(query.split("&")) if query else 0,
            "email_in_url": int(bool(re.search(r"[\w\.-]+@[\w\.-]+", url))),
            "url_shortened": int(any(token in url.lower() for token in SHORTENER_PATTERNS)),
        }
    )

    for name in PLACEHOLDER_FEATURES:
        features[name] = -1

    for col in columns:
        if col not in features:
            features[col] = -1

    return pd.DataFrame([features], columns=columns)


def format_prediction(prediction: int) -> str:
    return "Phishing" if int(prediction) == 1 else "Legitimate"


def get_probability_score(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])
    if hasattr(model, "decision_function"):
        return float(model.decision_function(X)[0])
    return float(np.nan)


def explain_shap(estimator, scaler, X: pd.DataFrame, feature_columns: list[str]) -> None:
    X_scaled = scaler.transform(X) if scaler is not None else X.values
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    raw_contributions = shap_values[0]
    shap_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "shap_value": raw_contributions,
            "abs_shap": np.abs(raw_contributions),
            "feature_value": X.iloc[0].values,
        }
    ).sort_values("abs_shap", ascending=False)

    st.subheader("SHAP Feature Contributions")
    st.write(
        "Positive SHAP values push the prediction toward phishing; negative values push it toward legitimate."
    )
    top_features = shap_df.head(12).sort_values("shap_value")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features["feature"], top_features["shap_value"], color=["#d62728" if v > 0 else "#1f77b4" for v in top_features["shap_value"]])
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title("Top feature contributions for this URL")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    st.pyplot(fig)

    st.markdown("---")
    st.dataframe(top_features[["feature", "feature_value", "shap_value"]].reset_index(drop=True))


def main():
    st.set_page_config(
        page_title="Phishing URL Classifier",
        page_icon=":shield:",
        layout="centered",
    )

    st.title("Phishing URL Detection Prototype")
    st.markdown(
        "This demo uses a trained phishing detection pipeline and provides real-time explainability with SHAP feature contributions."
    )

    _, scaler, estimator, feature_columns = load_model()

    url_input = st.text_input(
        "Enter a URL to classify:",
        value="http://example.com/login?user=123",
        placeholder="https://example.com/path?query=1",
    )

    if st.button("Classify URL"):
        if not url_input.strip():
            st.warning("Please enter a URL before classifying.")
            return

        X_input = extract_url_features(url_input, feature_columns)
        prediction = int(estimator.predict(scaler.transform(X_input) if scaler is not None else X_input)[0])
        score = get_probability_score(estimator, scaler.transform(X_input) if scaler is not None else X_input)
        label = format_prediction(prediction)

        st.success(f"Prediction: {label}")
        