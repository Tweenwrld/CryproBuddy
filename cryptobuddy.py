import nltk
import re
import requests
import json
import difflib
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

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
            analysis += f"{crypto} ({amount}): Trend = {data['price_trend']}, Sustainability = {data['sustainability_score']*10}/10, Risk = {data['risk_score']}, Volatility = {data['volatility']}\n"
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

    def find_crypto(self, name):
        """Fuzzy match crypto names."""
        name = name.capitalize()
        if name in self.crypto_data:
            return name
        # Support aliases (e.g., BTC → Bitcoin)
        aliases = {"BTC": "Bitcoin", "ETH": "Ethereum", "ADA": "Cardano", "SOL": "Solana", "BNB": "Binance Coin",
                   "DOT": "Polkadot", "AVAX": "Avalanche", "ATOM": "Cosmos", "ALGO": "Algorand", "NEAR": "NEAR Protocol"}
        if name in aliases:
            return aliases[name]
        # Fuzzy matching for misspellings
        matches = difflib.get_close_matches(name, self.crypto_data.keys(), n=1, cutoff=0.8)
        return matches[0] if matches else name

    def preprocess_query(self, query):
        """Tokenize and normalize user input, support multi-language."""
        tokens = word_tokenize(query.lower())
        for en_key, translations in keyword_translations.items():
            for lang, trans in translations.items():
                if trans in tokens:
                    tokens.append(en_key)
        return tokens, query.lower()

    def recommend_profitability(self):
        """Recommend based on profitability."""
        for crypto, data in self.crypto_data.items():
            if data["price_trend"] == "rising" and data["market_cap"] in ["high", "medium"]:
                return f"🚀 For profitability, I recommend {crypto}! It’s trending up with a {data['market_cap']} market cap."
        return "No highly profitable coins match your criteria right now."

    def recommend_sustainability(self):
        """Recommend based on sustainability."""
        best_crypto = max(self.crypto_data, key=lambda x: self.crypto_data[x]["sustainability_score"])
        if self.crypto_data[best_crypto]["sustainability_score"] > 7/10:
            return f"🌱 For sustainability, go with {best_crypto}! It’s eco-friendly with a sustainability score of {self.crypto_data[best_crypto]['sustainability_score']*10}/10."
        return "No highly sustainable coins found."

    def recommend_low_risk(self):
        """Recommend low-risk coins."""
        low_risk_coins = [crypto for crypto, data in self.crypto_data.items() if data["risk_score"] == "low"]
        if low_risk_coins:
            return f"🛡️ For low-risk investments, consider {', '.join(low_risk_coins)}!"
        return "No low-risk coins found."

    def recommend_staking(self):
        """Recommend coins for staking."""
        staking_coins = [crypto for crypto, data in self.crypto_data.items() if data["staking_available"]]
        if staking_coins:
            return f"💰 For staking, consider {', '.join(staking_coins)}! They offer rewards for holding."
        return "No staking coins found."

    def recommend_by_use_case(self, use_case):
        """Recommend coins by use case."""
        matching_coins = [crypto for crypto, data in self.crypto_data.items() if data["use_case"] == use_case]
        if matching_coins:
            return f"🔧 For {use_case}, consider {', '.join(matching_coins)}!"
        return f"No coins found for {use_case}."

    def recommend_new_coins(self):
        """Recommend recently launched coins."""
        new_coins = [crypto for crypto, data in self.crypto_data.items() if data["project_age"] <= 3]
        if new_coins:
            return f"🆕 New coins to watch: {', '.join(new_coins)}!"
        return "No new coins found."

    def compare_coins(self, coins):
        """Compare multiple coins across attributes."""
        comparison = "Comparison:\n"
        for crypto in coins:
            if crypto in self.crypto_data:
                data = self.crypto_data[crypto]
                comparison += f"{crypto}: Trend = {data['price_trend']}, Market Cap = {data['market_cap']}, Sustainability = {data['sustainability_score']*10}/10, Volatility = {data['volatility']}, Use Case = {data['use_case']}\n"
            else:
                comparison += f"{crypto}: Not found in database.\n"
        return comparison

    def analyze_sentiment(self, crypto):
        """Analyze sentiment of mock X posts."""
        crypto = self.find_crypto(crypto)
        if crypto in self.mock_x_posts:
            scores = [self.sid.polarity_scores(post)["compound"] for post in self.mock_x_posts[crypto]]
            avg_score = sum(scores) / len(scores)
            sentiment = "positive" if avg_score > 0.05 else "negative" if avg_score < -0.05 else "neutral"
            return f"📢 X sentiment for {crypto}: {sentiment} (based on recent posts)."
        return "No sentiment data available."

    def predict_price(self, crypto):
        """Simple rule-based price prediction."""
        crypto = self.find_crypto(crypto)
        if crypto in self.crypto_data:
            trend = self.crypto_data[crypto]["price_trend"]
            if trend == "rising":
                return f"📈 {crypto} is likely to keep rising short-term based on current trends."
            elif trend == "stable":
                return f"📊 {crypto} is stable, good for long-term holding."
            else:
                return f"📉 {crypto} is falling, consider waiting before investing."
        return "Crypto not found."

    def fetch_realtime_data(self, coin_id):
        """Fetch real-time price trend from CoinGecko API."""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id.lower()}/market_chart?vs_currency=usd&days=30"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            prices = data["prices"]
            avg_price = sum(price[1] for price in prices) / len(prices)
            latest_price = prices[-1][1]
            trend = "rising" if latest_price > avg_price else "stable" if latest_price == avg_price else "falling"
            return f"📈 Real-time 30-day trend for {coin_id.capitalize()}: {trend}"
        except Exception as e:
            return f"Error fetching data: {str(e)}"

    def log_feedback(self, query, response, feedback):
        """Store user feedback."""
        self.feedback_log[query] = {"response": response, "feedback": feedback}
        return "Thanks for your feedback! It helps me improve."

    def handle_query(self, query):
        """Process user query and return response."""
        tokens, normalized_query = self.preprocess_query(query)
        response = ""

        # Handle portfolio queries
        if "add" in normalized_query and "portfolio" in normalized_query:
            match = re.search(r"add\s+(\d+\.?\d*)\s+(\w+)", normalized_query)
            if match:
                amount, crypto = match.groups()
                response = self.portfolio.add_holding(crypto, float(amount))

        elif "portfolio" in normalized_query:
            response = self.portfolio.analyze_portfolio()

        # Handle profitability, sustainability, risk, staking, and use case
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

        # Handle comparison
        elif "compare" in normalized_query:
            coins = [self.find_crypto(coin) for coin in re.findall(r"\b[A-Za-z]+\b", normalized_query) if coin.capitalize() in self.crypto_data or self.find_crypto(coin) in self.crypto_data]
            if len(coins) >= 2:
                response = self.compare_coins(coins)
            else:
                response = "Please specify at least two valid coins to compare (e.g., 'Compare Bitcoin and Ethereum')."

        # Handle sentiment, prediction, and real-time data
        elif "sentiment" in normalized_query:
            for crypto in self.crypto_data:
                if self.find_crypto(crypto.lower()) in normalized_query:
                    response = self.analyze_sentiment(crypto)
                    break
            else:
                response = "Please specify a crypto for sentiment analysis (e.g., 'Bitcoin sentiment')."
        elif "predict" in normalized_query or "future" in normalized_query:
            for crypto in self.crypto_data:
                if self.find_crypto(crypto.lower()) in normalized_query:
                    response = self.predict_price(crypto)
                    break
            else:
                response = "Please specify a crypto for price prediction (e.g., 'Predict Bitcoin')."
        elif "real-time" in normalized_query:
            for crypto in self.crypto_data:
                if self.find_crypto(crypto.lower()) in normalized_query:
                    response = self.fetch_realtime_data(crypto.lower())
                    break
            else:
                response = "Please specify a crypto for real-time data (e.g., 'Real-time Bitcoin')."

        # Handle educational queries
        elif "what is" in normalized_query:
            for term, explanation in crypto_tips.items():
                if term in normalized_query:
                    response = f"📚 {explanation}"
                    break
            else:
                response = "I don’t have an explanation for that term. Try 'What is market cap?' or 'What is staking?'"

        # Handle feedback
        elif "feedback" in normalized_query:
            match = re.search(r"feedback\s+(\w+)", normalized_query)
            if match:
                feedback = match.group(1)
                last_query = list(self.feedback_log.keys())[-1] if self.feedback_log else None
                if last_query:
                    response = self.log_feedback(last_query, self.feedback_log.get(last_query, {}).get("response", ""), feedback)
                else:
                    response = "No previous query to provide feedback on."

        # Default response
        else:
            response = "Hmm, I’m not sure what you’re asking. Try something like ‘Which crypto is trending up?’, ‘Compare Bitcoin and Ethereum’, or ‘What is staking?’"

        return f"{response}\n{self.disclaimer}"

def main():
    bot = CryptoBuddy(crypto_db)
    print("Welcome to CryptoBuddy! 🌍🚀 Your ultimate crypto assistant!")
    print("Examples: 'Which crypto is trending up?', 'Add 0.1 BTC to portfolio', 'Compare Bitcoin and Solana', 'What’s the sentiment for Ethereum?', 'What is DeFi?', 'Welche Krypto ist am nachhaltigsten?'")
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