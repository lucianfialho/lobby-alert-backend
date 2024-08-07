from flask import Flask, request, jsonify
import redis
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)
load_dotenv()

# Configuração do cliente Redis
redis_url = os.getenv("REDIS_URL")
redis_client = redis.Redis.from_url(redis_url, decode_responses=True)


@app.route("/analyze", methods=["POST"])
def analyze_metrics():
    data = request.get_json()
    if not data or "profiles" not in data:
        return jsonify({"error": "No data provided"}), 400

    try:
        level_dict = process_profiles(data["profiles"])
        risk_label = calculate_risk(level_dict)
        return jsonify({"risk": risk_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def process_profiles(profiles):
    level_dict = {}
    for profile in profiles:
        data = extract_and_process_profile_data(profile)
        if data:
            level = data["level"]
            if level not in level_dict:
                level_dict[level] = []
            level_dict[level].append(data)
    return level_dict


def calculate_risk(level_dict):
    thresholds = {"High": 0.20, "Medium": 0.10}  # Define risk thresholds
    overall_risk_score = 0
    total_players = 0
    for level, players in level_dict.items():
        if len(players) > 1:
            df = pd.DataFrame(players)
            outliers = apply_isolation_forest(df)
            num_outliers = sum(1 for result in outliers if result["outlier"] == -1)
            num_players = len(players)
            total_players += num_players
            risk_score = num_outliers / num_players
            overall_risk_score += risk_score * num_players

    overall_risk = (overall_risk_score / total_players) if total_players else 0
    if overall_risk >= thresholds["High"]:
        return "High"
    elif overall_risk >= thresholds["Medium"]:
        return "Medium"
    else:
        return "Low"


def extract_and_process_profile_data(profile):
    metrics = profile.get("metrics", {})
    stats = metrics.get("stats", {})
    data = {
        "level": metrics.get("level"),
        **{
            k: stats.get(k, 0)
            for k in [
                "assists",
                "clutches",
                "deaths",
                "firstKills",
                "headshots",
                "kddiff",
                "kdr",
                "adr",
            ]
        },
    }
    return data


def apply_isolation_forest(df):
    clf = IsolationForest(n_estimators=100, contamination="auto")
    df["outlier"] = clf.fit_predict(df.drop(columns=["level"]))
    return df.to_dict(orient="records")


if __name__ == "__main__":
    app.run(port=5000)
