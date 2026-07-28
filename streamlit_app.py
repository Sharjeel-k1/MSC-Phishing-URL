import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# --- CONFIGURATION ---
MODEL_PATH = Path(__file__).parent / "final_phishing_model.pkl"

CHARACTER_MAP = {
    "dot": ".", "hyphen": "-", "underline": "_", "slash": "/", 
    "questionmark": "?", "equal": "=", "at": "@", "and": "&", 
    "exclamation": "!", "space": " ", "tilde": "~", "comma": ",", 
    "plus": "+", "asterisk": "*", "hashtag": "#", "dollar": "$", "percent": "%",
}

SHORTENER_PATTERNS = [
    "bit.ly", "tinyurl", "goo.gl", "ow.ly", "t.co", "is.gd", "buff.ly", "adf.ly",
]

TRUSTED_BRANDS = {
    "google.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "paypal.com",
    "github.com",
    "office.com",
    "outlook.com",
    "netflix.com",
    "dropbox.com",
    "facebook.com",
    "linkedin.com",
    "instagram.com",
    "twitter.com",
    "bankofamerica.com",
}

LOOKALIKE_REPLACEMENTS = {
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "!": "i",
    "$": "s",
}

# CYBERSECURITY FIX: Defensive Baseline Imputation
# We assign normal, legitimate values to offline metrics to prevent the model 
# from classifying a site as malicious simply because offline data was unavailable.
BENIGN_OFFLINE_FEATURES = {
    "time_response": 0.5,             
    "domain_spf": 1,                  
    "asn_ip": 13335,                  
    "time_domain_activation": 3650,   
    "time_domain_expiration": 365,    
    "qty_ip_resolved": 1,             
    "qty_nameservers": 2,             
    "qty_mx_servers": 1,              
    "ttl_hostname": 3600,             
    "tls_ssl_certificate": 1,         
    "qty_redirects": 0,               
    "url_google_index": 0,            
    "domain_google_index": 0
}


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
    # Safely strip malicious whitespace and enforce scheme
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    parsed = urlparse(url)
    domain = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""
    file_name = path.split("/")[-1] if path else ""
    directory = "/".join(path.split("/")[1:-1]) if "/" in path else ""

    features = {}
    field_values = {
        "url": url, "domain": domain, "directory": directory,
        "file": file_name, "params": query,
    }

    # Extract lexical features
    for field, value in field_values.items():
        for token, symbol in CHARACTER_MAP.items():
            features[f"qty_{token}_{field}"] = count_char(value, symbol)

    # Extract structural and domain features
    features.update(
        {
            "qty_tld_url": len([part for part in domain.split(".") if part]) - 1 if domain else 0,
            "length_url": len(url),
            "qty_vowels_domain": sum(domain.lower().count(v) for v in "aeiou"),
            "domain_length": len(domain),
            "domain_in_ip": int(is_ip_address(domain)),
            "server_client_domain": 0,
            "directory_length": len(directory),
            "file_length": len(file_name),
            "params_length": len(query),
            "tld_present_params": int("." in query),
            "qty_params": len(query.split("&")) if query else 0,
            "email_in_url": int(bool(re.search(r"[\w\.-]+@[\w\.-]+", url))),
            "url_shortened": int(any(token in url.lower() for token in SHORTENER_PATTERNS)),
        }
    )

    # --- FIX 4: Missing Component Override ---
    # If the URL has no parameters, set all param features to -1 (Absent) instead of 0
    if not query:
        features["params_length"] = -1
        features["tld_present_params"] = -1
        features["qty_params"] = -1
        for token in CHARACTER_MAP.keys():
            features[f"qty_{token}_params"] = -1
            
    # If the URL has no directory, set all directory features to -1
    if not directory:
        features["directory_length"] = -1
        for token in CHARACTER_MAP.keys():
            features[f"qty_{token}_directory"] = -1
            
    # If the URL has no file extension, set all file features to -1
    if not file_name:
        features["file_length"] = -1
        for token in CHARACTER_MAP.keys():
            features[f"qty_{token}_file"] = -1

    # Apply secure baselines for offline features
    for name, safe_value in BENIGN_OFFLINE_FEATURES.items():
        features[name] = safe_value

    # Fill any remaining missing columns required by the model with 0
    for col in columns:
        if col not in features:
            features[col] = 0

    return pd.DataFrame([features], columns=columns)


def format_prediction(prediction: int) -> str:
    return "Phishing" if int(prediction) == 1 else "Legitimate"


def normalize_lookalike_domain(domain: str) -> str:
    domain = (domain or "").lower().strip(".")
    if not domain:
        return ""
    normalized = []
    for char in domain:
        normalized.append(LOOKALIKE_REPLACEMENTS.get(char, char))
    return "".join(normalized)


def detect_lookalike_domain(url: str) -> bool:
    url = (url or "").strip()
    if not url:
        return False
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().strip(".")
    if not hostname:
        return False

    normalized = normalize_lookalike_domain(hostname)
    if not normalized:
        return False

    for brand in TRUSTED_BRANDS:
        if normalized == brand and hostname != brand:
            return True

    return False


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
    
    # Safely normalize SHAP output dimensions to prevent backend crashes
    if isinstance(shap_values, list) and len(shap_values) >= 2:
        shap_values = shap_values[1]
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    raw_contributions = np.array(shap_values[0]).flatten()
    feature_vals = np.array(X.iloc[0].values).flatten()

    shap_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "shap_value": raw_contributions,
            "abs_shap": np.abs(raw_contributions),
            "feature_value": feature_vals,
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

    st.title("🛡️ Phishing URL Detection Prototype")
    st.markdown(
        "This SOC dashboard uses a trained classification pipeline to analyze structural URL threats and provides real-time explainability with SHAP feature contributions."
    )

    try:
        _, scaler, estimator, feature_columns = load_model()
    except FileNotFoundError:
        st.error(f"Model file not found! Please ensure '{MODEL_PATH.name}' is in the same directory as this script.")
        st.stop()

    url_input = st.text_input(
        "Enter a target URL to classify:",
        placeholder="example-suspicious-login.com/auth",
    )

    if st.button("Classify URL"):
        if not url_input.strip():
            st.warning("Please enter a URL before classifying.")
            return

        with st.spinner("Analyzing structural anomalies..."):
            X_input = extract_url_features(url_input, feature_columns)
            
            X_processed = scaler.transform(X_input) if scaler is not None else X_input
            
            prediction = int(estimator.predict(X_processed)[0])
            score = get_probability_score(estimator, X_processed)
            label = format_prediction(prediction)

            lookalike_detected = detect_lookalike_domain(url_input)
            if lookalike_detected:
                prediction = 1
                score = 0.99
                label = format_prediction(prediction)

            st.divider()
            if prediction == 1:
                st.error(f"🚨 **Threat Detected:** The model classified this URL as **{label}**.")
                if lookalike_detected:
                    st.warning("Lookalike domain pattern detected: the hostname uses character substitutions that resemble a legitimate brand.")
            else:
                st.success(f"✅ **Clear:** The model classified this URL as **{label}**.")
                
            if not np.isnan(score):
                st.info(f"**Threat Probability Score:** {score:.2%}")

            explain_shap(estimator, scaler, X_input, feature_columns)

if __name__ == "__main__":
    main()