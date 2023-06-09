#Prerequisites: pip install openai pandas

import openai
import pandas as pd
import csv
import os
import yaml
import json
import time
import collections
import datetime
from openai.error import RateLimitError

# Replace with your OpenAI API key
openai.api_key = ""

def calculate_lirads(i,data):
    # Access the "features" dictionary of the "0" key in the "data" dictionary
    features = data[i]["features"]
    #Major features add
    major_add = sum(features.get(key, 0) for key in ["NPWO", "ECAP", "TG"])
    #LR-M
    sum_lrm = sum(features.get(key, 0) for key in ["RimAPHE", "PW", "DCE", "TgtDR", "TgtTPHBP", "Infilt", "MkdDR", "Nec", "SevIsch"])
    lrm = 1 if sum_lrm > 0 else 0
    #Benign ancillary features add
    sum_b = sum(features.get(key, 0) for key in ["SizStbl", "Reduction", "ParBlood", "UnV", "Iron", "MkT2", "HBPi"])
    b_add = 1 if sum_b > 0 else 0
    #Malignant ancillary add
    sum_m = sum(features.get(key, 0) for key in ["DiscUS", "sTG", "DR", "mT2", "Cor", "NoFat", "NoIron", "TPlow", "HBPlow", "nonCAP", "NiN", "Msc", "Blood", "Fat"])
    m_add = 1 if sum_m > 0 else 0
    # Calculate lirads value
    lirads = ("TIV" if features["TIV"] == 1 else "M" if lrm == 1 else None)
    # If lirads not TIV nor M
    if lirads is None:
        aphe = features["NonrimAPHE"]
        size = features["size"]
        if size == "":
            lirads = "NC"
        else:
            lirads = (
                5 if aphe and ((major_add >= 2 and size >= 10) or (major_add == 1 and size >= 20) or (major_add == 1 and 10 <= size < 20 and features["ECAP"] == 0)) else
                4 if (aphe and ((major_add >= 2 and size < 10) or (major_add == 1 and (10 <= size < 20 and features["ECAP"] == 1) or size < 10)
                                or (major_add == 0 and size >= 20))) or (not aphe and (major_add == 2 or (major_add == 1 and size >= 20))) else 3
            )
            if not ((b_add == 0 and m_add == 1 and lirads == 4) or (b_add == 0 and m_add ==1 and lirads == 5)):
                lirads += m_add - b_add
    # Add lirads to data dictionary
    data[i]["features"]["lirads"] = lirads

def process_text(report, template):
    prompt = template.replace("(input)", report)
    prompt = prompt.replace("(glsry)", glossary)
    for i in range(2):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4", 
                messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.01) #0.1
            return response.choices[0].message.content
        except RateLimitError as e:
            # Rate limit error encountered, retry after 10 seconds
            print(f"Rate limit error encountered. Retrying in 30 seconds... ({e})")
            time.sleep(30)
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(30)
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            time.sleep(30)
    return ""

# Load the YAML file into a Python object
#Real test with prompts_gpt4_twopart_mod.yml
with open("prompts_gpt4_twopart_mod.yml", "r", encoding="utf-8") as file:
    templates = yaml.load(file, Loader=yaml.FullLoader)

glossary = templates["glossary"]
system = templates["system_prompt"]

# Read the report variables
reports_df = pd.read_csv("reports.csv", encoding="utf-8")
reports = reports_df["report"].tolist()

# Prepare the response and extract variables
responses = []
extracts = []
outputs = {i: {} for i in range(len(reports))}

# Full loop with save on each loop
features_list = []

for i, report in enumerate(reports[0:], start=0):
    report_responses = []
    report_extract = process_text(report, templates["prompts"]["summarize"])
    outputs[i]["summary"] = json.loads(report_extract)
    response = process_text(report_extract, templates["prompts"]["features"])
    outputs[i]["features"] = json.loads(response)
    calculate_lirads(i,outputs)
    # Append the current features to the features_list
    features_list.append(outputs[i]["features"])

    # Create a DataFrame from the list of dictionaries
    responses_df = pd.DataFrame(features_list, index=[k for k in range(i+1) if "features" in outputs[k]])

    # Save the response and extract variables as CSV files
    responses_df.to_csv("responses.csv", index=False)

    # Check if JSON file exists
    json_file = "responses.json"
    if os.path.exists(json_file):
        with open(json_file, "r") as infile:
            existing_data = json.load(infile)
        existing_data[i] = outputs[i]  # Update only the current key in the existing data
        data_to_write = existing_data
    else:
        data_to_write = outputs

    # Write response to JSON file
    with open(json_file, "w+") as outfile:
        json.dump(data_to_write, outfile, indent=4)
    
    print("Report generated for " + str(i) + " at " + str(datetime.datetime.now()))

# create a list of dictionaries containing the "features" data
features_list = [outputs[k]["features"] for k in outputs if "features" in outputs[k]]
# create a DataFrame from the list of dictionaries
responses_df = pd.DataFrame(features_list, index=[k for k in outputs if "features" in outputs[k]])
# Save the response and extract variables as CSV files
responses_df.to_csv("responses_test_all_final.csv", index=False)

### MERGE columns
r_merge = responses_df
# lrm-t
lrm_t = ["RimAPHE","PW","DCE","TgtDR","TgtTPHBP"]
r_merge["LRMT"] = r_merge[lrm_t].sum(axis=1)
r_merge["LRMT"] = r_merge["LRMT"].apply(lambda x: 1 if x >=1 else 0)
r_merge = r_merge.drop(lrm_t,axis=1)
# lrm-nt
lrm_nt = ["Infilt","MkdDR","Nec","SevIsch"]
r_merge["LRMNT"] = r_merge[lrm_nt].sum(axis=1)
r_merge["LRMNT"] = r_merge["LRMNT"].apply(lambda x: 1 if x >=1 else 0)
r_merge = r_merge.drop(lrm_nt,axis=1)
# baf
baf = ["SizStbl","Reduction","ParBlood","UnV","Iron","MkT2","HBPi"]
r_merge["BAF"] = r_merge[baf].sum(axis=1)
r_merge["BAF"] = r_merge["BAF"].apply(lambda x: 1 if x >=1 else 0)
r_merge = r_merge.drop(baf,axis=1)
# maf
maf = ["DiscUS","sTG","DR","mT2","Cor","NoFat","NoIron","TPlow","HBPlow","nonCAP","NiN","Msc","Blood","Fat"]
r_merge["MAF"] = r_merge[maf].sum(axis=1)
r_merge["MAF"] = r_merge["MAF"].apply(lambda x: 1 if x >=1 else 0)
r_merge = r_merge.drop(maf,axis=1)
# column reorder
r_merge = r_merge.reindex(columns=["size", "location", "risk_factors", "LRMT", "LRMNT", "TIV", "NonrimAPHE","NPWO","ECAP","TG","BAF","MAF"])

r_merge.to_csv("responses_merged.csv", index=False)
