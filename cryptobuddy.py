import nltk
import re
import requests
import json
import difflib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
import speech_recognition as sr
import pyttsx3

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Expanded crypto dataset
crypto_db = {
    "Bitcoin": {
        "price_trend": "rising",
        "market_cap": "high",
        "energy_use": "high",
        "sustainability_score": 3/10,
        "trading_volume": "high",
        "risk_score": "low",
        "volatility": "medium",
        "project_age": 15,
        "use_case": "payments",
        "staking_available": False
    },
    "Ethereum": {
        "price_trend": "stable",
        "market_cap": "high",
        "energy_use": "medium",
        "sustainability_score": 6/10,
        "trading_volume": "high",
        "risk_score": "low",
        "volatility": "medium",
        "project_age": 9,
        "use_case": "smart contracts",
        "staking_available": True
    },
    "Cardano": {
        "price_trend": "rising",
        "market_cap": "medium",
        "energy_use": "low",
        "sustainability_score": 8/10,
        "trading_volume": "medium",
        "risk_score": "medium",
        "volatility": "high",
        "project_age": 6,
        "use_case": "smart contracts",
        "staking_available": True
    },
    "Solana": {
        "price_trend": "rising",
        "market_cap": "high",
        "energy_use": "low",
        "sustainability_score": 7/10,
        "trading_volume": "high",
        "risk_score": "medium",
        "volatility": "high",
        "project_age": 4,
        "use_case": "smart contracts",
        "staking_available": True
    },
    "Binance Coin": {
        "price_trend": "stable",
        "market_cap": "high",
        "energy_use": "medium",
        "sustainability_score": 5/10,
        "trading_volume": "high",
        "risk_score": "low",
        "volatility": "medium",
        "project_age": 7,
        "use_case": "exchange token",
        "staking_available": True
    },
    "Polkadot": {
        "price_trend": "falling",
        "market_cap": "medium",
        "energy_use": "low",
        "sustainability_score": 7/10,
        "trading_volume": "medium",
        "risk_score": "medium",
        "volatility": "high",
        "project_age": 4,
        "use_case": "interoperability",
        "staking_available": True
    },
    "Avalanche": {
        "price_trend": "rising",
        "market_cap": "medium",
        "energy_use": "low",
        "sustainability_score": 7/10,
        "trading_volume": "medium",
        "risk_score": "medium",
        "volatility": "high",
        "project_age": 3,
        "use_case": "smart contracts",
        "staking_available": True
    },
    "Cosmos": {
        "price_trend": "stable",
        "market_cap": "medium",
        "energy_use": "low",
        "sustainability_score": 7/10,
        "trading_volume": "medium",
        "risk_score": "medium",
        "volatility": "medium",
        "project_age": 5,
        "use_case": "interoperability",
        "staking_available": True
    },
    "Algorand": {
        "price_trend": "falling",
        "market_cap": "medium",
        "energy_use": "low",
        "sustainability_score": 8/10,
        "trading_volume": "medium",
        "risk_score": "medium",
        "volatility": "high",
        "project_age": 5,
        "use_case": "smart contracts",
        "staking_available": True
    },
    "NEAR Protocol": {
        "price_trend": "rising",
        "market_cap": "medium",
        "energy_use": "low",
        "sustainability_score": 7/10,
        "trading_volume": "medium",
        "risk_score": "medium",
        "volatility": "high",
        "project_age": 3,
        "use_case": "smart contracts",
        "staking_available": True
    }
}

# Expanded keyword translations
keyword_translations = {
    "profit": {"es": "ganancia", "fr": "profit", "de": "gewinn", "zh": "利润"},
    "sustainable": {"es": "sostenible", "fr": "durable", "de": "nachhaltig", "zh": "可持续"},
    "trending": {"es": "tendencia", "fr": "tendance", "de": "trending", "zh": "趋势"},
    "safe": {"es": "seguro", "fr": "sûr", "de": "sicher", "zh": "安全"},
    "staking": {"es": "staking", "fr": "staking", "de": "staking", "zh": "质押"},
    "defi": {"es": "defi", "fr": "defi", "de": "defi", "zh": "去中心化金融"},
    "new": {"es": "nuevo", "fr": "nouveau", "de": "neu", "zh": "新"},
    "compare": {"es": "comparar", "fr": "comparer", "de": "vergleichen", "zh": "比较"}
}

# Expanded educational tips
crypto_tips = {
    "market cap": "Market cap is the total value of a crypto’s circulating supply. Higher market cap often means more stability!",
    "sustainability": "Sustainable cryptos use less energy, like Cardano’s proof-of-stake system, reducing environmental impact.",
    "staking": "Staking involves locking crypto to support a blockchain network and earn rewards, like interest.",
    "defi": "DeFi (Decentralized Finance) offers financial services like lending or trading without intermediaries.",
    "volatility": "Volatility measures price fluctuations. High volatility means bigger price swings, higher risk."
}

# Mock on-chain data
on_chain_data = {
    "Bitcoin": {"tx_volume": "high", "active_addresses": 900000},
    "Ethereum": {"tx_volume": "high", "active_addresses": 500000},
    "Cardano": {"tx_volume": "medium", "active_addresses": 200000},
    "Solana": {"tx_volume": "high", "active_addresses": 300000},
    "Binance Coin": {"tx_volume": "medium", "active_addresses": 150000},
    "Polkadot": {"tx_volume": "medium", "active_addresses": 100000},
    "Avalanche": {"tx_volume": "medium", "active_addresses": 80000},
    "Cosmos": {"tx_volume": "low", "active_addresses": 60000},
    "Algorand": {"tx_volume": "low", "active_addresses": 70000},
    "NEAR Protocol": {"tx_volume": "medium", "active_addresses": 90000}
}

class Portfolio:
    def __init__(self):
        self.holdings = self.load_portfolio()

    def add_holding(self, crypto, amount):
        crypto = self.find_crypto(crypto)
        if crypto in crypto_db:
            self.holdings[crypto] = self.holdings.get(crypto, 0) + float(amount)
            self.save_portfolio()
            return f"Added {amount} {crypto} to your portfolio!"
        return "Crypto not found in database."

    def analyze_portfolio(self):
        if not self.holdings:
            return "Your portfolio is empty. Add some cryptos!"
        analysis = "Portfolio Analysis:\n"
        for crypto, amount in self.holdings.items():
            data = crypto_db[crypto]
            analysis += f"{crypto} ({amount}): Trend: {data['price_trend']}, Sustainability: {data['sustainability_score']*10}/10, Risk: {data['risk_score']}, Volatility: {data['volatility']}\n"
        return analysis

    def save_portfolio(self):
        with open("portfolio.json", "w") as f:
            json.dump(self.holdings, f)

    def load_portfolio(self):
        try:
            with open("portfolio.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def find_crypto(self, name):
        name = name.capitalize()
        if name in crypto_db:
            return name
        aliases = {"BTC": "Bitcoin", "ETH": "Ethereum", "ADA": "Cardano", "SOL": "Solana", "BNB": "Binance Coin",
                   "DOT": "Polkadot", "AVAX": "Avalanche", "ATOM": "Cosmos", "ALGO": "Algorand", "NEAR": "NEAR Protocol"}
        if name in aliases:
            return aliases[name]
        matches = difflib.get_close_matches(name, crypto_db.keys(), n=1, cutoff=0.8)
        return matches[0] if matches else name

class UserProfile:
    def __init__(self):
        self.preferences = {"risk_tolerance": "medium", "use_case": None}

    def set_preference(self, key, value):
        self.preferences[key] = value
        return f"Set {key} to {value}."

    def get_recommendation(self, crypto_data):
        if self.preferences["risk_tolerance"] == "low":
            return [crypto for crypto, data in crypto_data.items() if data["risk_score"] == "low"]
        elif self.preferences["use_case"]:
            return [crypto for crypto, data in crypto_data.items() if data["use_case"] == self.preferences["use_case"]]
        return list(crypto_data.keys())

class CryptoBuddy:
    def __init__(self, crypto_data):
        self.crypto_data = crypto_data
        self.portfolio = Portfolio()
        self.sid = SentimentIntensityAnalyzer()
        self.feedback_log = {}
        self.disclaimer = "⚠️ Crypto investments are risky—always do your own research!"
        self.mock_x_posts = {
            "Bitcoin": ["BTC to the moon! 🚀", "Bitcoin is crashing, beware!"],
            "Ethereum": ["ETH is stable, great for staking!", "ETH fees are too high."],
            "Cardano": ["Cardano’s eco-friendly approach is the future! 🌱"],
            "Solana": ["Solana is super fast!", "SOL fees are low but volatile."],
            "Binance Coin": ["BNB powers Binance!", "BNB has limited use cases."],
            "Polkadot": ["DOT connects blockchains!", "Polkadot is complex for newbies."],
            "Avalanche": ["AVAX is scalable!", "Avalanche needs more adoption."],
            "Cosmos": ["Cosmos enables interoperability!", "ATOM is under the radar."],
            "Algorand": ["Algorand is green!", "ALGO has low fees but less hype."],
            "NEAR Protocol": ["NEAR is developer-friendly!", "NEAR is still growing."]
        }
        self.user_profile = UserProfile()
        self.on_chain_data = on_chain_data

    def find_crypto(self, name):
        name = name.capitalize()
        if name in self.crypto_data:
            return name
        aliases = {"BTC": "Bitcoin", "ETH": "Ethereum", "ADA": "Cardano", "SOL": "Solana", "BNB": "Binance Coin",
                   "DOT": "Polkadot", "AVAX": "Avalanche", "ATOM": "Cosmos", "ALGO": "Algorand", "NEAR": "NEAR Protocol"}
        if name in aliases:
            return aliases[name]
        matches = difflib.get_close_matches(name, self.crypto_data.keys(), n=1, cutoff=0.8)
        return matches[0] if matches else name

    def preprocess_query(self, query):
        tokens = word_tokenize(query.lower())
        for en_key, translations in keyword_translations.items():
            for lang, trans in translations.items():
                if trans.lower() in tokens:
                    tokens.append(en_key)
        # Extract potential coin names
        coins = [self.find_crypto(token) for token in tokens if self.find_crypto(token) in self.crypto_data]
        return tokens, query.lower(), coins

    def recommend_profitability(self):
        coins = self.user_profile.get_recommendation(self.crypto_data)
        for crypto in coins:
            if self.crypto_data[crypto]["price_trend"] == "rising" and self.crypto_data[crypto]["market_cap"] in ["high", "medium"]:
                return f"🚀 For profitability, I recommend {crypto}! It’s trending up with a {self.crypto_data[crypto]['market_cap']} market cap."
        return "No highly profitable coins match your criteria right now."

    def recommend_sustainability(self):
        coins = self.user_profile.get_recommendation(self.crypto_data)
        best_crypto = max(coins, key=lambda x: self.crypto_data[x]["sustainability_score"])
        if self.crypto_data[best_crypto]["sustainability_score"] > 7/10:
            return f"🌱 For sustainability, I recommend {best_crypto}! It’s eco-friendly with a sustainability score of {self.crypto_data[best_crypto]['sustainability_score']*10}/10."
        return "No highly sustainable coins found."

    def recommend_low_risk(self):
        low_risk_coins = [crypto for crypto, data in self.crypto_data.items() if data["risk_score"] == "low"]
        if low_risk_coins:
            return f"🛡️ For low-risk investments, consider {', '.join(low_risk_coins)}!"
        return "No low-risk coins found."

    def recommend_staking(self):
        staking_coins = [crypto for crypto, data in self.crypto_data.items() if data["staking_available"]]
        if staking_coins:
            return f"💰 For staking, consider {', '.join(staking_coins)}! They offer rewards for holding."
        return "No staking coins found."

    def recommend_by_use_case(self, use_case):
        matching_coins = [crypto for crypto, data in self.crypto_data.items() if data["use_case"] == use_case.lower()]
        if matching_coins:
            return f"🔧 For {use_case}, consider {', '.join(matching_coins)}!"
        return f"No coins found for {use_case}."

    def recommend_new_coins(self):
        new_coins = [crypto for crypto, data in self.crypto_data.items() if data["project_age"] <= 3]
        if new_coins:
            return f"🆕 New coins to watch: {', '.join(new_coins)}!"
        return "No new coins found."

    def compare_coins(self, coins):
        comparison = "Comparison:\n"
        for crypto in coins:
            if crypto in self.crypto_data:
                data = self.crypto_data[crypto]
                comparison += f"{crypto}: Trend: {data['price_trend']}, Market Cap: {data['market_cap']}, Sustainability: {data['sustainability_score']*10}/10, Volatility: {data['volatility']}, Use Case: {data['use_case']}\n"
            else:
                comparison += f"{crypto}: Not found in database.\n"
        return comparison

    def analyze_sentiment(self, crypto):
        crypto = self.find_crypto(crypto)
        if crypto in self.mock_x_posts:
            scores = [self.sid.polarity_scores(post)["compound"] for post in self.mock_x_posts[crypto]]
            avg_score = sum(scores) / len(scores)
            sentiment = "positive" if avg_score > 0.05 else "negative" if avg_score < -0.05 else "neutral"
            return f"📢 Sentiment for {crypto}: {sentiment} (based on recent posts)."
        return "No sentiment data available."

    def predict_price_ml(self, coin_id):
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id.lower()}/market_chart?vs_currency=usd&days=30"
            response = requests.get(url).json()
            prices = response["prices"]
            X = np.array([i for i in range(len(prices))]).reshape(-1, 1)
            y = np.array([price[1] for price in prices])
            model = LinearRegression().fit(X, y)
            next_day = np.array([[len(prices)]])
            predicted_price = model.predict(next_day)[0]
            return f"📈 Predicted next-day price for {coin_id.capitalize()}: ${predicted_price:.2f} (based on linear regression)."
        except Exception as e:
            return f"Error predicting price: {str(e)}"

    def fetch_news(self, crypto):
        crypto = self.find_crypto(crypto)
        try:
            url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={crypto.lower()}&lTs=3600"
            response = requests.get(url).json()
            news = response.get("Data", [])[:3]
            if not news:
                return f"No recent news found for {crypto}."
            summary = f"📰 Recent news for {crypto}:\n"
            for article in news:
                summary += f"- {article['title']} ({article['source']})\n"
            return summary
        except Exception as e:
            return f"Error fetching news: {str(e)}"

    def fetch_realtime_data(self, coin_id):
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id.lower()}"
            response = requests.get(url).json()
            price = response["market_data"]["current_price"]["usd"]
            volume = response["market_data"]["total_volume"]["usd"]
            return f"📊 Real-time data for {coin_id.capitalize()}: Price: ${price:.2f}, 24h Volume: ${volume:.2f}"
        except Exception as e:
            return f"Error fetching real-time data: {str(e)}"

    def recommend_on_chain(self):
        best_crypto = max(self.on_chain_data, key=lambda x: self.on_chain_data[x]["active_addresses"])
        return f"📊 For network activity, {best_crypto} has high transaction volume and {self.on_chain_data[best_crypto]['active_addresses']} active addresses."

    def simulate_transaction(self, crypto, amount, action):
        crypto = self.find_crypto(crypto)
        if crypto not in self.crypto_data:
            return "Crypto not found."
        if action == "buy":
            self.portfolio.add_holding(crypto, amount)
            return f"Bought {amount} {crypto} (mock transaction, 1% fee applied)."
        elif action == "sell" and crypto in self.portfolio.holdings:
            if self.portfolio.holdings[crypto] >= amount:
                self.portfolio.holdings[crypto] -= amount
                self.portfolio.save_portfolio()
                return f"Sold {amount} {crypto} (mock transaction, 1% fee applied)."
            return "Insufficient balance."
        return "Invalid action."

    def voice_query(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=3)
                query = recognizer.recognize_google(audio)
                print(f"You said: {query}")
                response = self.handle_query(query)
                engine = pyttsx3.init()
                engine.say(response)
                engine.runAndWait()
                return response
            except sr.UnknownValueError:
                return "Sorry, I didn’t understand that."
            except sr.RequestError:
                return "Voice service unavailable."
            except Exception as e:
                return f"Voice error: {str(e)}"

    def log_feedback(self, query, response, feedback):
        self.feedback_log[query] = {"response": response, "feedback": feedback}
        return f"Thanks for the {feedback} feedback on '{query}'!"

    def generate_chart(self, metric, coins=None):
        if coins is None:
            coins = list(self.crypto_data.keys())
        else:
            coins = [self.find_crypto(coin) for coin in coins if self.find_crypto(coin) in self.crypto_data]
        
        if not coins:
            return "No valid coins selected."

        metric_map = {
            "sustainability": {"label": "Sustainability Score (/10)", "key": "sustainability_score", "scale": 10},
            "market_cap": {"label": "Market Cap (Relative)", "key": "market_cap", "scale": None},
            "volatility": {"label": "Volatility (Relative)", "key": "volatility", "scale": None}
        }
        
        if metric not in metric_map:
            return "Invalid metric for chart. Try 'sustainability', 'market_cap', or 'volatility'."

        labels = coins
        data = []
        for coin in coins:
            value = self.crypto_data[coin][metric_map[metric]["key"]]
            if metric == "market_cap":
                value = {"high": 3, "medium": 2, "low": 1}.get(value, 0)
            elif metric == "volatility":
                value = {"high": 3, "medium": 2, "low": 1}.get(value, 0)
            else:
                value = value * metric_map[metric]["scale"]
            data.append(value)

        chart_config = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": metric_map[metric]["label"],
                    "data": data,
                    "backgroundColor": ["#FF6384", "#36A2EB", "#4BC0C0", "#FFCE56", "#E7E9ED", 
                                       "#9966FF", "#FF9F40", "#C9CBCF", "#4CAF50", "#F44336"],
                    "borderColor": ["#FF6384", "#36A2EB", "#4BC0C0", "#FFCE56", "#E7E9ED", 
                                    "#9966FF", "#FF9F40", "#C9CBCF", "#4CAF50", "#F44336"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": metric_map[metric]["scale"] if metric_map[metric]["scale"] else None
                    }
                },
                "plugins": {
                    "legend": {"display": True},
                    "title": {
                        "display": True,
                        "text": f"Cryptocurrency {metric_map[metric]['label']}"
                    }
                }
            }
        }
        return f"Chart.js config for LMS submission:\n```chartjs\n{json.dumps(chart_config, indent=2)}\n```"

    def generate_ascii_chart(self, metric, coins=None):
        if coins is None:
            coins = list(self.crypto_data.keys())
        else:
            coins = [self.find_crypto(coin) for coin in coins if self.find_crypto(coin) in self.crypto_data]
        
        if not coins:
            return "No valid coins selected for chart."

        metric_map = {
            "sustainability": {"key": "sustainability_score", "scale": 10},
            "market_cap": {"key": "market_cap", "scale": None},
            "volatility": {"key": "volatility", "scale": None}
        }
        
        if metric not in metric_map:
            return "Invalid metric for chart. Try 'sustainability', 'market_cap', or 'volatility'."

        data = []
        max_label_len = max(len(coin) for coin in coins)
        for coin in coins:
            value = self.crypto_data[coin][metric_map[metric]["key"]]
            if metric == "market_cap":
                value = {"high": 3, "medium": 2, "low": 1}.get(value, 0)
            elif metric == "volatility":
                value = {"high": 3, "medium": 2, "low": 1}.get(value, 0)
            else:
                value = value * metric_map[metric]["scale"]
            data.append((coin, value))

        max_value = max(value for _, value in data) or 1
        chart = f"ASCII Chart for {metric.capitalize()}:\n"
        for value in range(int(max_value), 0, -1):
            line = f"{value:2}|"
            for _, coin_value in data:
                line += "█" if coin_value >= value else " "
                line += " "
            chart += line + "\n"
        chart += "  +" + "--" * len(data) + "\n"
        chart += "  " + " ".join(coin[:2] for coin, _ in data) + "\n"
        chart += "\nLegend:\n" + "\n".join(f"{coin.ljust(max_label_len)}: {value}" for coin, value in data)
        return chart

    def handle_query(self, query):
        tokens, normalized_query, coins = self.preprocess_query(query)
        response = ""

        if normalized_query == "voice":
            response = self.voice_query()

        elif "add" in normalized_query and "portfolio" in normalized_query:
            match = re.search(r"add\s+(\d+\.?\d*)\s+(\w+)", normalized_query)
            if match:
                amount, crypto = match.groups()
                response = self.portfolio.add_holding(crypto, float(amount))

        elif "buy" in normalized_query:
            match = re.search(r"buy\s+(\d+\.?\d*)\s+(\w+)", normalized_query)
            if match:
                amount, crypto = match.groups()
                response = self.simulate_transaction(crypto, float(amount), "buy")

        elif "sell" in normalized_query:
            match = re.search(r"sell\s+(\d+\.?\d*)\s+(\w+)", normalized_query)
            if match:
                amount, crypto = match.groups()
                response = self.simulate_transaction(crypto, float(amount), "sell")

        elif "portfolio" in normalized_query:
            response = self.portfolio.analyze_portfolio()

        elif "chart" in normalized_query:
            metric = None
            if "sustainability" in normalized_query:
                metric = "sustainability"
            elif "market cap" in normalized_query:
                metric = "market_cap"
            elif "volatility" in normalized_query:
                metric = "volatility"
            
            if metric:
                response = self.generate_ascii_chart(metric, coins if coins else None)
                response += "\n" + self.generate_chart(metric, coins if coins else None)
            else:
                response = "Please specify a metric for the chart (e.g., 'Show sustainability chart')."

        elif "profit" in normalized_query or "trending" in normalized_query or "growth" in normalized_query:
            response = self.recommend_profitability()

        elif "sustainab" in normalized_query or "eco" in normalized_query or "green" in normalized_query:
            response = self.recommend_sustainability()

        elif "safe" in normalized_query or "low risk" in normalized_query:
            response = self.recommend_low_risk()

        elif "staking" in normalized_query:
            response = self.recommend_staking()

        elif "defi" in normalized_query:
            response = self.recommend_by_use_case("smart contracts")

        elif "payment" in normalized_query:
            response = self.recommend_by_use_case("payments")

        elif "new" in normalized_query or "recent" in normalized_query:
            response = self.recommend_new_coins()

        elif "network" in normalized_query or "on-chain" in normalized_query:
            response = self.recommend_on_chain()

        elif "compare" in normalized_query or len(coins) >= 2:  # Implicit comparison
            if len(coins) >= 2:
                response = self.compare_coins(coins)
                if "compare" not in normalized_query:
                    response += f"\nTip: You can also say 'Compare {coins[0]} and {coins[1]}' for this!"
            else:
                response = "Please specify at least two valid coins to compare (e.g., 'Compare Bitcoin and Ethereum')."

        elif "sentiment" in normalized_query:
            if coins:
                response = self.analyze_sentiment(coins[0])
            else:
                response = "Please specify a crypto for sentiment analysis (e.g., 'Bitcoin sentiment')."

        elif "predict" in normalized_query or "future" in normalized_query:
            if coins:
                response = self.predict_price_ml(coins[0].lower())
            else:
                response = "Please specify a crypto for price prediction (e.g., 'Predict Bitcoin')."

        elif "news" in normalized_query:
            if coins:
                response = self.fetch_news(coins[0])
            else:
                response = "Please specify a crypto for news (e.g., 'Bitcoin news')."

        elif "real-time" in normalized_query:
            if coins:
                response = self.fetch_realtime_data(coins[0].lower())
            else:
                response = "Please specify a crypto for real-time data (e.g., 'Real-time Bitcoin')."

        elif "set risk" in normalized_query:
            match = re.search(r"set risk\s+(\w+)", normalized_query)
            if match:
                response = self.user_profile.set_preference("risk_tolerance", match.group(1))

        elif "set use case" in normalized_query:
            match = re.search(r"set use case\s+([\w\s]+)", normalized_query)
            if match:
                response = self.user_profile.set_preference("use_case", match.group(1).strip())

        elif "what is" in normalized_query:
            for term, explanation in crypto_tips.items():
                if term in normalized_query:
                    response = f"📚 {explanation}"
                    break
            else:
                response = "I don’t have an explanation for that term. Try 'What is market cap?' or 'What is staking?'"

        elif "feedback" in normalized_query:
            match = re.search(r"feedback\s+(\w+)", normalized_query)
            if match:
                feedback = match.group(1)
                last_query = list(self.feedback_log.keys())[-1] if self.feedback_log else None
                if last_query:
                    response = self.log_feedback(last_query, self.feedback_log.get(last_query, {}).get("response", ""), feedback)
                else:
                    response = "No previous query to provide feedback on."

        else:
            response = f"Hmm, I’m not sure what you meant. Try 'Compare {', '.join(coins)}' or 'Predict {coins[0]}' if you meant those!" if coins else "Try something like ‘Show sustainability chart’, ‘Predict Bitcoin’, or ‘What is DeFi?’"

        return f"{response}\n{self.disclaimer}"

def main():
    bot = CryptoBuddy(crypto_db)
    print("Welcome to CryptoBuddy! 🌍🚀 Your ultimate crypto assistant!")
    print("Examples: 'Show sustainability chart', 'Compare Bitcoin and Solana', 'Predict Bitcoin', 'Solana news', 'Buy 0.1 BTC', 'Set risk to low', 'voice' for voice mode, 'What is staking?'")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Thanks for chatting with CryptoBuddy! Stay savvy! 😎")
            break
        response = bot.handle_query(query)
        print(f"CryptoBuddy: {response}\n")

if __name__ == "__main__":
    main()