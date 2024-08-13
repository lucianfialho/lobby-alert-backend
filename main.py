from flask import Flask, request, jsonify
import redis
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.ensemble import IsolationForest
import concurrent.futures
from flask_cors import CORS
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
load_dotenv()

# Configuração do cliente Redis
redis_url = os.getenv("REDIS_URL")
redis_client = redis.Redis.from_url(redis_url, decode_responses=True)


@app.route("/analyze", methods=["POST"])
def analyze_metrics():
    logger.info("Recebida requisição para /analyze")
    data = request.get_json()
    if not data or "profiles" not in data:
        logger.warning("Nenhum dado fornecido ou chave 'profiles' ausente")
        return jsonify({"error": "No data provided"}), 400

    try:
        level_dict = process_profiles(data["profiles"])
        risk_label = calculate_and_store_risk(level_dict)
        logger.info(f"Risco calculado: {risk_label}")
        return jsonify({"risk": risk_label})
    except Exception as e:
        logger.error(f"Erro durante a análise: {str(e)}")
        return jsonify({"error": str(e)}), 500


def process_profiles(profiles):
    logger.info("Processando perfis")
    level_dict = {}
    for profile in profiles:
        data = extract_and_process_profile_data(profile)
        if data:
            level = data["level"]
            if level not in level_dict:
                level_dict[level] = []
            level_dict[level].append(data)
    logger.info(f"Perfis processados por nível: {level_dict.keys()}")
    return level_dict


def calculate_and_store_risk(level_dict):
    logger.info("Calculando risco para os perfis")
    thresholds = {"High": 0.20, "Medium": 0.10}
    min_players_required = 10
    overall_risk_score = 0
    total_players = 0

    def process_level(level, players):
        logger.info(f"Processando nível {level} com {len(players)} jogadores")
        redis_players = fetch_players_from_redis(level)
        combined_players = players + redis_players
        logger.info(
            f"Total de jogadores após combinar com Redis: {len(combined_players)}"
        )
        if len(combined_players) >= min_players_required:
            df = pd.DataFrame(combined_players)
            outliers = apply_isolation_forest(df)
            num_outliers = sum(1 for result in outliers if result["outlier"] == -1)
            num_players = len(combined_players)
            risk_score = num_outliers / num_players
            save_players_to_redis(level, players)
            logger.info(
                f"Nível {level}: {num_outliers} outliers encontrados em {num_players} jogadores"
            )
            return (num_outliers, num_players, risk_score)
        else:
            logger.warning(
                f"Nível {level} não tem jogadores suficientes para aplicar o Isolation Forest"
            )
            return (0, 0, 0)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda args: process_level(*args), level_dict.items())

    for num_outliers, num_players, risk_score in results:
        if num_players:
            total_players += num_players
            overall_risk_score += risk_score * num_players

    overall_risk = (overall_risk_score / total_players) if total_players else 0
    logger.info(f"Risco geral calculado: {overall_risk}")
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
    logger.debug(f"Dados processados para o jogador: {data}")
    return data


def fetch_players_from_redis(level):
    logger.info(f"Buscando jogadores do nível {level} no Redis")
    pattern = f"user:{level}:*"
    keys = redis_client.keys(pattern)

    with redis_client.pipeline() as pipe:
        for key in keys:
            pipe.get(key)
        players_data = pipe.execute()

    players = [eval(player) for player in players_data if player]
    logger.info(f"{len(players)} jogadores encontrados no Redis para o nível {level}")
    return players


def save_players_to_redis(level, players):
    expiration_time = 7 * 24 * 60 * 60  # 7 dias em segundos
    for player in players:
        steam_id = player.get("id")
        if steam_id:
            redis_key = f"user:{level}:{steam_id}"
            redis_client.setex(redis_key, expiration_time, str(player))
    logger.info(f"Jogadores do nível {level} salvos no Redis")


def apply_isolation_forest(df):
    logger.info("Aplicando Isolation Forest")
    clf = IsolationForest(
        n_estimators=50, contamination="auto"
    )  # Reduzi para 50 estimadores para otimização
    df["outlier"] = clf.fit_predict(df.drop(columns=["level"]))
    logger.info(
        f"Isolation Forest aplicado. Outliers identificados: {sum(df['outlier'] == -1)}"
    )
    return df.to_dict(orient="records")


if __name__ == "__main__":
    logger.info("Iniciando servidor Flask")
    app.run(port=5000)
