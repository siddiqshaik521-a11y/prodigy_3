import pandas as pd
import requests
import io

def download_dataset():
    url = "https://github.com/Prodigy-InfoTech/data-science-datasets/raw/refs/heads/main/Task%203/bank-additional/bank-additional-full.csv"
    print(f"Downloading dataset from {url}...")
    
    response = requests.get(url)
    if response.status_code == 200:
        # The dataset uses ';' as a separator
        df = pd.read_csv(io.StringIO(response.text), sep=';')
        df.to_csv("bank_marketing.csv", index=False)
        print("Dataset saved as bank_marketing.csv")
        return df
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")
        return None

if __name__ == "__main__":
    download_dataset()
