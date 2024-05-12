import sys
import re
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QMessageBox, QHBoxLayout
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import GPT2Tokenizer
from bs4 import BeautifulSoup
from ollama import Client

class TextSummarizer(QWidget):
    def __init__(self):
        super().__init__()
        self.client = Client(host='http://localhost:11434')
        self.init_ui()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.connect_to_server()

    def init_ui(self):
        self.setWindowTitle("Text Summarizer and Tweet Generator")

        # Main layout
        main_layout = QVBoxLayout()

        # User input interface
        self.input_label = QLabel("Enter text, a link to an article, or a YouTube video URL:")
        main_layout.addWidget(self.input_label)

        self.input_entry = QTextEdit()
        main_layout.addWidget(self.input_entry)

        # Custom Prompt interface
        self.custom_prompt_label = QLabel("Custom Prompt (Optional):")
        main_layout.addWidget(self.custom_prompt_label)

        self.custom_prompt_entry = QTextEdit()
        main_layout.addWidget(self.custom_prompt_entry)

        # Buttons for actions
        buttons_layout = QHBoxLayout()
        self.summarize_button = QPushButton("Summarize")
        self.summarize_button.clicked.connect(lambda: self.process_input("summary"))
        buttons_layout.addWidget(self.summarize_button)

        self.generate_tweet_button = QPushButton("Generate Tweet")
        self.generate_tweet_button.clicked.connect(lambda: self.process_input("tweet"))
        buttons_layout.addWidget(self.generate_tweet_button)

        main_layout.addLayout(buttons_layout)

        # Result display
        self.result_label = QLabel("Result:")
        main_layout.addWidget(self.result_label)

        self.result_text = QTextEdit()
        main_layout.addWidget(self.result_text)

        self.setLayout(main_layout)

    def connect_to_server(self):
        # Check if the required model is available. If not, pull it.
        model_name = "llama3:8b"
        try:
            self.client.show(model_name)
        except:
            self.client.pull(model_name)

    def process_input(self, prompt_type):
        input_text = self.input_entry.toPlainText().strip()
        if input_text.startswith("http://") or input_text.startswith("https://"):
            if "youtube.com" in input_text or "youtu.be" in input_text:
                try:
                    input_text = self.fetch_transcript_from_youtube(input_text)
                except ValueError as e:
                    QMessageBox.warning(self, "Error", str(e))
                    return
            else:
                try:
                    input_text = self.fetch_text_from_url(input_text)
                except ValueError as e:
                    QMessageBox.warning(self, "Error", str(e))
                    return

        if input_text:
            try:
                generated_text = self.generate_response_from_text(input_text, prompt_type)
                self.result_text.setText(generated_text)
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
        else:
            QMessageBox.warning(self, "Input Error", "Please enter text, a link to an article, or a YouTube video URL.")

    def generate_response_from_text(self, text, prompt_type):
        custom_prompt = self.custom_prompt_entry.toPlainText().strip()
        if custom_prompt:
            prompt = custom_prompt
        else:
            if prompt_type == "tweet":
                prompt = "Generate an opinionated Twitter response based on the provided text."
            elif prompt_type == "summary":
                prompt = "Generate a concise summary from the provided text. write five bullets points on the key info and then write a summary expanding on those points into a 500 word essay."

        try:
            chunks = self.chunk_text(text, max_length=2000)
            chunk_prompts = [f"{prompt}\n\n{chunk}" for chunk in chunks]
            
            response = self.client.chat(
                model="llama3:8b",
                messages=[{"role": "user", "content": "\n\n".join(chunk_prompts)}]
            )
            generated_text = response.get('message', {}).get('content', '').strip()

            return generated_text
        except Exception as e:
            raise RuntimeError(f"Error generating response: {str(e)}") from e

    def fetch_text_from_url(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the article content based on common HTML tags or classes
            article_content = soup.find('article')
            if not article_content:
                article_content = soup.find('div', class_='article-body')
            if not article_content:
                article_content = soup.find('div', class_='entry-content')

            if article_content:
                text = article_content.get_text(separator=' ')
                return text
            else:
                raise ValueError("Unable to find article content in the HTML.")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error fetching text from the link: {str(e)}") from e

    def fetch_transcript_from_youtube(self, url):
        try:
            video_id = self.extract_video_id(url)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join(entry["text"] for entry in transcript)
            return transcript_text
        except Exception as e:
            raise ValueError(f"Error fetching transcript from YouTube: {str(e)}") from e

    def extract_video_id(self, url):
        youtube_regex = (
            r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([^&\?]+)'
        )
        match = re.match(youtube_regex, url)
        if match:
            return match.group(4)
        else:
            raise ValueError("Invalid YouTube URL")

    def chunk_text(self, text, max_length=2000):
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = min(start + max_length, len(tokens))
            chunk = self.tokenizer.decode(tokens[start:end])
            chunks.append(chunk)
            start = end

        return chunks

if __name__ == "__main__":
    app = QApplication(sys.argv)
    summarizer = TextSummarizer()
    summarizer.show()
    sys.exit(app.exec_())
