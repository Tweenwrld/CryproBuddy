# CryptoBuddy

Welcome to CryptoBuddy! This is a rule-based chatbot I built to provide cryptocurrency investment advice based on profitability, sustainability, staking opportunities, and more. I designed it to be beginner-friendly yet robust, handling a wide range of prompts to assist crypto enthusiasts in making informed decisions.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
  - [Google Colab](#option-1-google-colab)
  - [Local IDE (VS Code, PyCharm)](#option-2-local-ide-vs-code-pycharm)
  - [Jupyter Notebook](#option-3-jupyter-notebook)
- [How to Use](#how-to-use)
- [Example Interaction](#example-interaction)
- [Accepted Prompts](#accepted-prompts)
- [Capabilities](#capabilities)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
I implemented CryptoBuddy in Python using rule-based logic to analyze a comprehensive dataset of 10 cryptocurrencies. I designed it to offer recommendations for profitability (price trends, market cap), sustainability (energy use, sustainability score), staking, and specific use cases (e.g., DeFi, payments). I integrated features like portfolio tracking, sentiment analysis of mock X posts, real-time CoinGecko API data, and multi-language support. The chatbot mimics AI decision-making with modular code, making it scalable and engaging.

---

## Features
- **Recommendations:** I programmed CryptoBuddy to suggest coins based on profitability, sustainability, low risk, staking, or use cases (e.g., smart contracts, payments).
- **Portfolio Tracking:** I added the ability to add and analyze crypto holdings, persisted to a JSON file.
- **Comparisons:** I enabled comparisons of multiple coins across metrics like trend, sustainability, and volatility.
- **Sentiment Analysis:** I implemented sentiment analysis of mock X posts for community sentiment (positive, neutral, negative).
- **Price Predictions:** I included rule-based short-term price forecasts.
- **Real-Time Data:** I fetch 30-day price trends via CoinGecko API.
- **Multi-Language Support:** I made sure it handles queries in English, Spanish, French, German, and Chinese.
- **Educational Tips:** I included explanations for terms like "market cap," "staking," and "DeFi."
- **Feedback Loop:** I log user feedback to simulate iterative improvement.
- **Ethical Disclaimer:** Every response includes "Crypto investments are risky—always do your own research!"

---

## Installation & Setup

### Option 1: Google Colab (Cloud-Based, Beginner-Friendly)
I found Google Colab ideal for quick setup without local installation. Here's how to run it:
1. Open Google Colab and create a new notebook.
2. Install dependencies:
   ```python
   !pip install nltk==3.8.1 requests==2.31.0
   import nltk
   nltk.download('punkt')
   nltk.download('vader_lexicon')
   nltk.download('averaged_perceptron_tagger')
   ```
3. Copy the contents of `cryptobuddy.py` into a code cell and run it.
4. **Note:** For portfolio persistence, mount Google Drive and update the portfolio path:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Modify Portfolio class to save to '/content/drive/MyDrive/portfolio.json'
   ```

### Option 2: Local IDE (VS Code, PyCharm)
I also tested CryptoBuddy locally for better control and GitHub integration. Follow these steps:
1. Ensure Python 3.8+ is installed:
   ```bash
   python --version
   ```
2. (Recommended) Set up a virtual environment:
   ```bash
   python -m venv cryptobuddy_env
   # Linux/Mac
   source cryptobuddy_env/bin/activate
   # Windows
   cryptobuddy_env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('vader_lexicon')
   nltk.download('averaged_perceptron_tagger')
   ```
5. Run the project:
   ```bash
   python cryptobuddy.py
   ```

### Option 3: Jupyter Notebook
I found Jupyter Notebook great for interactive testing:
1. Install Jupyter:
   ```bash
   pip install jupyter
   ```
2. Install dependencies (as above).
3. Start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Create a new notebook, paste `cryptobuddy.py`, and run the cells.

---

## How to Use
- Run `cryptobuddy.py` (or the Colab/Jupyter notebook).
- The program displays a welcome message and example prompts.
- Enter queries at the `You:` prompt (e.g., "Which crypto is trending up?").
- CryptoBuddy processes the query using rule-based logic and NLTK for tokenization, responding with recommendations or information.
- Type `exit` to quit.

---

## Example Interaction
```
Welcome to CryptoBuddy! 🌍🚀 Your ultimate crypto assistant!
Examples: 'Which crypto is trending up?', 'Add 0.1 BTC to portfolio', 'Compare Bitcoin and Solana', 'What's the sentiment for Ethereum?', 'What is DeFi?', 'Welche Krypto ist am nachhaltigsten?'
Type 'exit' to quit.

You: Compare Bitcoin and Solana
CryptoBuddy: Comparison:
  Bitcoin: Trend = rising, Market Cap = high, Sustainability = 3.0/10, Volatility = medium, Use Case = payments
  Solana: Trend = rising, Market Cap = high, Sustainability = 7.0/10, Volatility = high, Use Case = smart contracts
  ⚠️ Crypto investments are risky—always do your own research!

You: Which crypto is best for staking?
CryptoBuddy: 💰 For staking, consider Ethereum, Cardano, Solana, Binance Coin, Polkadot, Avalanche, Cosmos, Algorand, NEAR Protocol! They offer rewards for holding.
  ⚠️ Crypto investments are risky—always do your own research!

You: exit
Thanks for chatting with CryptoBuddy! Stay savvy! 😎
```

---

## Accepted Prompts
I expanded CryptoBuddy to handle a wide range of prompts for versatility:

- **Recommendations:**
  - Profitability: "Which crypto is trending up?", "What's good for growth?"
  - Sustainability: "What's the most sustainable coin?", "Which crypto is eco-friendly?"
  - Low Risk: "What's a safe crypto?", "Which coin has low risk?"
  - Staking: "Which crypto is best for staking?"
  - Use Cases: "Best crypto for DeFi", "What's good for payments?"
  - New Coins: "What are the newest coins?", "Recent cryptos?"
- **Portfolio:**
  - Add: "Add 0.1 BTC to portfolio", "Add 10 Solana"
  - View: "Show my portfolio", "What's in my portfolio?"
- **Comparisons:**
  - "Compare Bitcoin and Solana", "Bitcoin vs Ethereum vs Cardano"
- **Sentiment:**
  - "What's the sentiment for Ethereum?", "How's Bitcoin doing on X?"
- **Predictions:**
  - "Predict Bitcoin", "What's the future of Solana?"
- **Real-Time Data:**
  - "Real-time Bitcoin", "What's Solana's trend?"
- **Education:**
  - "What is market cap?", "What is staking?", "What is DeFi?"
- **Feedback:**
  - "Feedback good" (after a query)
- **Multi-Language:**
  - "Welche Krypto ist am nachhaltigsten?" (German), "Cuál es la cripto más sostenible?" (Spanish)

I also implemented fuzzy matching to handle misspellings (e.g., "Bitcon" → Bitcoin) and aliases (e.g., "BTC" → Bitcoin).

---

## Capabilities
CryptoBuddy is a versatile crypto assistant with the following capabilities:
- **Data Analysis:** I analyze a dataset of 10 cryptocurrencies across metrics like price trend, market cap, sustainability, volatility, and use case.
- **Personalization:** I track user portfolios and persist data to `portfolio.json`.
- **Social Insights:** I evaluate mock X post sentiment for each coin.
- **Real-Time Data:** I fetch 30-day price trends via CoinGecko API.
- **Education:** I explain key crypto terms for beginners.
- **Global Accessibility:** I support queries in English, Spanish, French, German, and Chinese.
- **Scalability:** My modular code allows easy addition of coins, attributes, or features (e.g., SQLite integration).

---

## Troubleshooting
I encountered a few challenges during development and testing:
- **NLTK Resource Errors:**
  - *Issue:* "Resource punkt not found."
  - *Fix:* Run:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')
    ```
- **CoinGecko API Failures:**
  - *Issue:* "Error fetching data" due to rate limits or connectivity issues.
  - *Fix:* Ensure internet access. Test API:
    ```python
    import requests
    print(requests.get("https://api.coingecko.com/api/v3/ping").status_code)  # Should print 200
    ```
  - If rate-limited, CryptoBuddy falls back to the static dataset.
- **Unrecognized Prompts:**
  - *Issue:* Queries like "What's Bitcon?" weren't recognized initially.
  - *Fix:* I added fuzzy matching and alias support (e.g., "BTC" → Bitcoin).
- **Portfolio Persistence in Colab:**
  - *Issue:* `portfolio.json` isn't saved in Colab's temporary storage.
  - *Fix:* Mount Google Drive or run locally for persistence.
- **Case Sensitivity:**
  - *Issue:* Queries like "add 0.1 btc" failed.
  - *Fix:* I normalized inputs to lowercase and added robust matching.
- **Debugging:**
  - For debugging, I recommend adding print statements in `preprocess_query`:
    ```python
    def preprocess_query(self, query):
        tokens, normalized_query = word_tokenize(query.lower()), query.lower()
        print(f"Tokens: {tokens}")
        return tokens, normalized_query
    ```

---

## Contributing
I welcome contributions to make CryptoBuddy even better! Ideas include:
- Adding more coins to the database
- Integrating a real X API for sentiment analysis
- Adding charts (e.g., Chart.js for sustainability scores)
- Supporting more languages or advanced NLP

**To contribute:**
1. Fork the repository
2. Create a branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Added feature-name"`
4. Push and create a pull request: `git push origin feature-name`

---

## License
This project is licensed under the MIT License.

---

## Screenshots
Add screenshots of interactions here (e.g., comparison, staking, portfolio). 