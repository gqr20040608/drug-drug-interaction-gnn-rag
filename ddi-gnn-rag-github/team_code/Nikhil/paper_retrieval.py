import urllib.parse
import requests
from bs4 import BeautifulSoup

EUTILS_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUTILS_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def fetch_pmc_fulltext_interaction_paper(drug1: str, drug2: str, retmax: int = 5):
    query = (
        f'("{drug1}"[All Fields]) AND ("{drug2}"[All Fields]) '
        f'AND (interaction[All Fields] OR "drug interaction"[All Fields])'
    )
    term = urllib.parse.quote(query)
    esearch_url = f"{EUTILS_ESEARCH}?db=pmc&term={term}&retmode=json&retmax={retmax}"
    r = requests.get(esearch_url, timeout=30)
    r.raise_for_status()

    idlist = r.json().get("esearchresult", {}).get("idlist", [])
    if not idlist:
        return None

    efetch_url = f"{EUTILS_EFETCH}?db=pmc&id={','.join(idlist)}&retmode=xml"
    r = requests.get(efetch_url, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml-xml")
    article = soup.find("article")

    if not article:
        return None

    title_tag = article.find("article-title")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    body = article.find("body")

    if not body:
        full_text = ""
    else:
        paras = [p.get_text(" ", strip=True) for p in body.find_all("p")]
        full_text = "\n\n".join([p for p in paras if p])

    pmcid_numeric = idlist[0]
    pmcid = f"PMC{pmcid_numeric}"

    return {
        "pmcid": pmcid,
        "title": title,
        "full_text": full_text,
        "drug1": drug1,
        "drug2": drug2,
        "source": "pmc",
    }


# paper = fetch_pmc_fulltext_interaction_paper("ibuprofen", "warfarin", retmax=5)

# if paper:
#     print(paper["pmcid"])
#     print(paper["title"])
#     print(paper["full_text"][:1200])
#     with open("output.txt", "w") as f:
#         f.write(paper["full_text"])

# else:
#     print("No match found in PMC")
