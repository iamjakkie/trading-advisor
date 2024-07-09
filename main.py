import os
import fitz  # PyMuPDF
import torch
from transformers import pipeline, AutoTokenizer
from agents.strategy_agent import create_agent

def load_pdf_data(directory):
    data = ""
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            try:
                with fitz.open(file_path) as doc:
                    for page in doc:
                        data += page.get_text()
            except RuntimeError as e:
                print(f"Error reading {file_path}: {e}")
    return data

def load_text_data(directory):
    data = ""
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                data += file.read() + "\n"
    return data

def chunk_text(text, tokenizer, max_tokens=1024):
    tokens = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return chunks

# Load data
books_data = load_pdf_data('data/books')
articles_data = load_text_data('data/articles')

# Initialize tokenizer and summarization pipeline
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer, device=device)

# Chunk data to fit model's max token limit
book_chunks = chunk_text(books_data, tokenizer, max_tokens=1024)
article_chunks = chunk_text(articles_data, tokenizer, max_tokens=1024)

# Decode token chunks back to text
book_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in book_chunks]
article_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in article_chunks]

# Summarize chunks and concatenate results
books_summary = " ".join([summarizer(chunk, max_length=512, min_length=30, do_sample=False)[0]['summary_text'] for chunk in book_chunks])
articles_summary = " ".join([summarizer(chunk, max_length=512, min_length=30, do_sample=False)[0]['summary_text'] for chunk in article_chunks])

# Create the agent
agent = create_agent(books_summary, articles_summary)

# Simple trading environment
import gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)  # Buy or Sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.state = np.random.rand(10)
        self.done = False

    def reset(self):
        self.state = np.random.rand(10)
        self.done = False
        return self.state

    def step(self, action):
        reward = np.random.rand()
        self.state = np.random.rand(10)
        self.done = np.random.rand() > 0.95
        return self.state, reward, self.done, {}

# Initialize the environment
env = TradingEnv()

# Ask the agent for a strategy
strategy = agent.ask("What is a good trading strategy?")
print("Suggested Strategy:", strategy)

# Simulate a simple trading episode
state = env.reset()
total_reward = 0

while True:
    action = 1 if "buy" in strategy.lower() else 0
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break

print("Total Reward:", total_reward)