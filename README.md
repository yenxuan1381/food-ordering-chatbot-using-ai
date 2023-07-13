# Kayo-The-Bot

This repository contains the source code for a food ordering chatbot using AI named Kayo. Kayo is designed to intelligently interact with users, provide menu options, and handle food orders.

## Features

- Natural Language Processing: Kayo uses NLTK and TensorFlow to understand and respond to user queries.
- Menu Presentation: Kayo can present different menus to the user based on their requests.
- Order Handling: Kayo can handle food orders, calculate total price, and manage the ordering process.
- Telegram Integration: Kayo is connected to Telegram and uses it as the chat interface for interacting with users.

## Dependencies

- Python 3.6 or higher
- NLTK
- TensorFlow
- Telebot
- NumPy
- json
- pickle
- os

Run these commands in the termindal:

```
pip install pyTelegramBotAPI
pip install tensorflow
pip install nltk
```

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Uncomment `nltk.download('punkt')` and `nltk.download('wordnet')` in training.py
4. Run main.py.

## Code Structure

- main.py: This is the main script that runs the chatbot. It includes the logic for handling user queries, presenting menus, and managing orders.
- intents.json: This file contains predefined patterns and responses for the chatbot.
- menu.json: This file contains the menu items that the chatbot can present to the user.
- words.pkl and labels.pkl: These files are used for processing user queries.
- chatbot_model.h5: This is the trained model that the chatbot uses to understand user queries.

## Note

This bot uses the Telebot library and requires a valid Telegram bot token to function. Please replace the placeholder token in the main.py file with your own token.

```
bot = telebot.TeleBot('YOUR_TELEGRAM_BOT_TOKEN')
```

## Example Use

This is a regular use case of using Kayo to order a food.

![Example Screenshot](img/demo.jpeg)

## Contribution

Feel free to fork this repository, make changes, and submit pull requests. If you find any bugs or have any suggestions, please open an issue.

## License

This project is licensed under the MIT License.