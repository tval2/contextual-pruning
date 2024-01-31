import os
import pandas as pd

from datasets import load_dataset, load_from_disk, Dataset
from utils import load_json

QUESTION_DATASETS_PATH = "question_datasets"


def load_questions(file_name):
    return load_json(os.path.join(QUESTION_DATASETS_PATH, f"{file_name}.json"))


def make_text_datasets(seed=42, datasets_to_load=[]):
    datasets = {}

    # Wikipedia test
    if not datasets_to_load or "wiki_test" in datasets_to_load:
        datasets["wiki_test"] = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    #Medical - 15.5k rows - text (string)
    if not datasets_to_load or "medical" in datasets_to_load:
        datasets["medical"] = load_dataset("Laurent1/MedQuad-MedicalQnADataset_128tokens_max", split="train")

    #Legal - 11.8k rows - text (string), label (string), category (string) (https://huggingface.co/datasets/lexlms/legal_lama)
    # terms_dataset = load_dataset('lexlms/legal_lama', name='us_terms', split="test", trust_remote_code=True)
    # crimes_dataset = load_dataset('lexlms/legal_lama', name='us_crimes', split="test")
    # contracts_dataset = load_dataset('lexlms/legal_lama', name='contract_sections', split="test")
    # legal_dataset = concatenate_datasets([terms_dataset, crimes_dataset, contracts_dataset])
    # Loading dataset works on Google colab but not locally, so dataset included in repo
    if not datasets_to_load or "legal" in datasets_to_load:
        datasets["legal"] = load_from_disk("legal_dataset")

    #English to Taiwanese - 311k rows - en (string), ch (string)
    if not datasets_to_load or "translation" in datasets_to_load:
        datasets["translation"] = load_dataset("zetavg/coct-en-zh-tw-translations-twp-300k", split="train")

    #Skyrim Dataset - 34.4k rows - text (string)
    if not datasets_to_load or "skyrim" in datasets_to_load:
        url = "https://github.com/jd7h/sentiment-lexicon-skyrim/blob/master/data/skyrim_dialogue.csv?raw=true"
        df = pd.read_csv(url, delimiter="\t")
        df.columns =['FORMID', 'Character', 'N/A1', 'Scene Type', "N/A2", "Number", "text", "N/A3", "N/A4"]
        datasets["skyrim"] = Dataset.from_pandas(df)

    #Economics Articles Dataset - 6.3k rows - url(string), text (string)
    if not datasets_to_load or "economics" in datasets_to_load:
        url = "https://github.com/RiksEddy/tinymlFP/blob/main/economics_text_dataset_8500.csv?raw=true"
        df = pd.read_csv(url).loc[:,'URL':'Text'].rename(columns={"URL": "url", "Text": "text"}).dropna(subset=['text'])
        datasets["economics"] = Dataset.from_pandas(df)

    for name, dataset in datasets.items():
        datasets[name] = dataset.shuffle(seed=seed)
    
    return datasets


def make_question_datasets(datasets_to_load=[]):
    datasets = {}

    # https://boballey.org/Trinity2/Economics%20Test.pdf
    if not datasets_to_load and "economics" in datasets_to_load:
        datasets["economics"] = load_questions("basic_economics_dataset")
    
    # Based on translation dataset (code in utils)
    if not datasets_to_load and "translation" in datasets_to_load:
        datasets["translation"] = load_questions("en_tw_translation_dataset")

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10388568/
    if not datasets_to_load and "medical" in datasets_to_load:
        datasets["medical"] = load_questions("rad_onc_phys_dataset")
    
    # Based on skyrim dialogue (multiple choice best sentence completion)
    if not datasets_to_load and "skyrim" in datasets_to_load:
        datasets["skyrim"] = load_questions("skyrim_question_dataset")

    return datasets


def make_datasets(seed=42, datasets=[], question_datasets=[]):
    return (
        make_text_datasets(seed=seed, datasets_to_load=datasets),
        make_question_datasets(datasets_to_load=question_datasets)
    )
