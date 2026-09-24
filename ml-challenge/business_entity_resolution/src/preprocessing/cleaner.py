import re
import unicodedata
import pandas as pd

class TextCleaner:
    @staticmethod
    def strip_accents(text: str) -> str:
        if not isinstance(text, str) or pd.isna(text):
            return ""
        text = unicodedata.normalize('NFKD', text)
        return "".join([c for c in text if not unicodedata.combining(c)])

    @classmethod
    def clean(cls, text: str) -> str:
        text = cls.strip_accents(text).lower()
        text = text.replace("&", " and ").replace("@", " at ")
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['clean_name'] = df['business_name'].apply(TextCleaner.clean)
    df['clean_address'] = df['business_address'].apply(TextCleaner.clean)
    df['clean_country'] = df['country'].fillna('').apply(TextCleaner.clean)
    
    df['search_text'] = (
        df['clean_country'] + " " + 
        df['clean_name'] + " " + 
        df['clean_address']
    ).str.strip()
    return df