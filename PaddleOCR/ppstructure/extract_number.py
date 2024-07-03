from predict_system import process
import os
from openai import OpenAI
import re
import pandas as pd
import textdistance

def output_ocr(output_ocr):
    output_ocr = output_ocr[0]
    # List of keywords to search for
    keywords = ["heineken", "tiger", "bia viet", "larue", "bivina", "edelweiss", "strongbow"]

    # Initialize a dictionary to hold the counts
    counts = {keyword: 0 for keyword in keywords}

    # Count the occurrences of each keyword
    for result in output_ocr["res"]:
        text = result["text"].lower()
        for keyword in keywords:
            distance = textdistance.levenshtein.normalized_similarity(keyword, text)
            if distance > 0.5:
                counts[keyword] += 1

    ret_s = ""
    for keyword, count in counts.items():
        ret_s += f"{keyword}: {count}, "
    return ret_s

# if __name__ == "__main__":
#     img_path = []
#     report = []
#     list_path = os.listdir("bienhieu")

#     for pth in list_path:
#         path = os.path.join('bienhieu', pth)
#         img_path.append(path)
#         output_o = process(path)
#         report.append(output_ocr(output_o))
#         break
#     df = pd.DataFrame({'img_path': img_path, 'report': report})
#     df.to_csv('csv_output/output.csv', index=False)