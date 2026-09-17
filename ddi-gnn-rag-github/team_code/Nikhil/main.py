from embedd_and_retrieve_text import store_text_embedded_weaviate
from paper_retrieval import fetch_pmc_fulltext_interaction_paper
from summarize_interactions import summarize_interaction_effects
import pickle
import os

def get_set():
	if os.path.exists("pairs.pkl"):
		with open("pairs.pkl", "rb") as f:
			embedded_pairs = pickle.load(f)

	else:
		embedded_pairs = set()
	
	return embedded_pairs


def save_set(embedded_pairs):
	with open("pairs.pkl", "wb") as f:
		pickle.dump(embedded_pairs, f)


def get_summary_pipeline(drug1, drug2):
	embedded_pairs = get_set()
	drug1, drug2 = sorted([drug1, drug2])
	pair = (drug1, drug2)

	if pair not in embedded_pairs:
		paper = fetch_pmc_fulltext_interaction_paper(drug1, drug2)
		store_text_embedded_weaviate(paper["full_text"])
		embedded_pairs.add(pair)
	
	save_set()
	return summarize_interaction_effects(drug1, drug2)


res = get_summary_pipeline("ibuprofen", "warfarin")
print(res)
