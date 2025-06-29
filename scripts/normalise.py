# Package for normalising data

from html.parser import HTMLParser
import unicodedata
import pandas as pd

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ' '.join(self.fed)

def strip_html(text):
    html_stripper = MLStripper()
    html_stripper.feed(text)
    return html_stripper.get_data()

def normalise_accented_characters(text):
    if not isinstance(text, str):
        if pd.isnull(text):
            return ''
        text = str(text)

    normalized = unicodedata.normalize('NFKD', text)
    ascii_bytes = normalized.encode('ascii', 'ignore')
    return ascii_bytes.decode('ascii')