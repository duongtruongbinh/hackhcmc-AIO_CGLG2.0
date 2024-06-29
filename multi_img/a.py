import os
from dotenv import load_dotenv
load_dotenv()
url = os.environ.get("IC_URL")
print(url)
