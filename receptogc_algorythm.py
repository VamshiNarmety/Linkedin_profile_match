import json
import re
from keybert import KeyBERT
from flair.data import Sentence
from flair.models import SequenceTagger
import itertools
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Tuple
from pydantic import BaseModel
import time
from typing_extensions import Optional, TypedDict
import requests
from pydantic import BaseModel
from typing import Optional, Dict, List
import difflib
from deepface import DeepFace
from sentence_transformers import SentenceTransformer, util
from wordfreq import word_frequency
import spacy

with open("dataset.json","r") as f:
    data=json.load(f)


"""### Queries Code"""

pipeline1_data=[]
pipeline2_data=[]
pipeline3_data=[]

def is_valid(field):
    return field is not None and field != ""

for entry in data:
    if is_valid(entry.get("intro")):
        pipeline1_data.append(entry)
    else:
        if not is_valid(entry.get("company_industry")) and not is_valid(entry.get("company_size")):
            pipeline2_data.append(entry)
        else:
            pipeline3_data.append(entry)

with open("pipe_1.json", "w") as f:
    json.dump(pipeline1_data, f, indent=4)

with open("pipe_2.json", "w") as f:
    json.dump(pipeline2_data, f, indent=4)

with open("pipe_3.json", "w") as f:
    json.dump(pipeline3_data, f, indent=4)


# print("Pipeline 1:")
# print(json.dumps(pipeline1_data, indent=4, ensure_ascii=False))
# print("Pipeline 2:")
# print(json.dumps(pipeline2_data, indent=4, ensure_ascii=False))
# print("Pipeline 3:")
# print(json.dumps(pipeline3_data, indent=4, ensure_ascii=False))

"""### Queries Code"""

##Common functions for all pipelines

# Load models
kw_model = KeyBERT("all-MiniLM-L6-v2")
ner_tagger = SequenceTagger.load("ner")
spacy_nlp = spacy.load("en_core_web_sm")

def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\(.*?\)", "", name).strip()
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name

def extract_words_from_url(text):
    urls = re.findall(r'https?://[^\s\)]+', text)
    keywords = []

    for url in urls:
        url = url.split('?')[0]  # Remove query parameters like ?utm_source=revgenius
        domain_match = re.search(r'https?://(?:www\.)?([a-zA-Z0-9\-]+)\.', url)
        if domain_match:
            keywords.append(domain_match.group(1).lower())

    return list(dict.fromkeys(keywords))

def is_uncommon(word):
    return word_frequency(word.lower(), 'en') < 1e-5

def extract_uncommon_words(text):
    words = re.findall(r'\b[A-Za-z0-9\-]{3,}\b', text.lower())
    return list(dict.fromkeys([w for w in words if is_uncommon(w)]))

def extract_entities(text):
    doc = spacy_nlp(text)
    return list({ent.text for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT"}})

def extract_keywords(text, max_keywords=5):
    keywords = kw_model.extract_keywords(text, top_n=max_keywords, stop_words="english")
    return [kw[0] for kw in keywords]

def extract_from_intro(text):
    uncommon_words = extract_uncommon_words(text)
    ner_entities = extract_entities(text)
    keywords = extract_keywords(text)

    combined = list(dict.fromkeys(uncommon_words + ner_entities + keywords))
    return [w for w in combined if len(w.split()) <= 4]

def remove_designations(text):
    if not isinstance(text, str):
        return ""

    titles = [
        "founder", "cofounder", "co-founder", "ceo", "cto", "cfo", "coo",
        "chief executive officer", "chief technology officer", "chief financial officer",
        "chief operating officer", "director", "vp", "president", "partner", "owner",
        "principal", "manager", "lead", "head", "entrepreneur", "investor", "consultant",
        "advisor", "angel investor", "freelancer", "self-employed"
    ]

    text = re.sub(r'[-/|&@]', ' ', text)
    for prefix in ["ex-", "ex "]:
        for title in titles:
            text = re.sub(rf"\b{prefix}{re.escape(title)}\b", "", text, flags=re.IGNORECASE)

    pattern = r'\b(?:' + '|'.join(re.escape(title) for title in titles) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return re.sub(r'\s{2,}', ' ', text).strip()

def generate_query(person):
    name = clean_name(person.get("name", ""))
    intro_raw = str(person.get("intro", "") or "")
    intro_clean = remove_designations(intro_raw)

    # Check for URLs in either raw or cleaned intro
    if "http" in intro_raw or "http" in intro_clean:
        keywords = extract_words_from_url(intro_raw + " " + intro_clean)
    else:
        keywords = extract_from_intro(intro_clean)

    keywords = [k for k in keywords if k.lower() not in name.lower()]

    # De-duplicate case-insensitively while keeping first appearance
    seen = set()
    unique_keywords = []
    for k in keywords:
        lower_k = k.lower()
        if lower_k not in seen:
            seen.add(lower_k)
            unique_keywords.append(k)

    final_keywords = unique_keywords[:2]
    return " ".join([name] + final_keywords)

def process_people(data):
    for person in data:
        person["query"] = generate_query(person)
    return data

if __name__ == "__main__":
    input_file = "dataset.json"
    output_file = "pipe1_processed.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = process_people(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=4)

    print("All entries processed and saved to", output_file)

import json
import time
from typing_extensions import Optional, TypedDict
import requests

class GoogleCustomSearchResponse(TypedDict):
    link: str
    title: Optional[str]
    snippet: Optional[str]

class GoogleCustomSearch:
    def __init__(self):
        self.api_key = "AIzaSyBhoMDnst3_JoK5TziGClgS27bhX0dQNFk"
        self.search_engine_id = "c307652d3fbc44ec8"
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def _search(self, query: str, num=10, start=1) -> Optional[dict]:
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.search_engine_id,
            "num": num,
            "start": start,
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Google Search Failed: {response.status_code} {response.text}")
            return None

    def _parse_results(self, response) -> list[GoogleCustomSearchResponse]:
        if not response or "items" not in response:
            return []
        results = []
        for item in response["items"]:
            link = item.get("link", None)
            if link:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "link": link,
                        "snippet": item.get("snippet", ""),
                    }
                )
        return results

    def search(self, query: str, num=5, start=1) -> list[GoogleCustomSearchResponse]:
        response = self._search(query, num, start)
        if response:
            return self._parse_results(response)
        return []

def main():
    # Load personas from processed.json
    with open('pipe1_processed.json', 'r', encoding='utf-8') as f:
        personas = json.load(f)

    searcher = GoogleCustomSearch()
    results = {}

    for persona in personas:
        name = persona.get("name", "Unknown")
        query = persona.get("query")
        print(f"Searching for: {name} | Query: {query}")
        if not query:
            results[name] = []
            continue
        search_results = searcher.search(f'site:linkedin.com/in {query}', num=5)
        results[name] = search_results
        time.sleep(1)  # Be polite to the API

    # Save results to a new JSON file
    with open('pipe1_urls.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("Done! Results saved to pipe1_urls.json")

if __name__ == "__main__":
    main()

import json
import itertools
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Tuple
from pydantic import BaseModel

# --- Data Models ---
class ProfileData(BaseModel):
    name: Optional[str] = None
    profile_url: Optional[str] = None
    profile_image: Optional[str] = None
    about: Optional[str] = None

# --- Rotating User Agents ---
user_agents = [
    "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
    "LinkedInBot/1.0",
    "Twitterbot/1.0",
    "facebookexternalhit/1.1",
    "WhatsApp/2.0",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
]
user_agent_cycle = itertools.cycle(user_agents)

def mimic_bot_headers() -> str:
    """Mimic bot headers by cycling through user agents"""
    return next(user_agent_cycle)

# --- LinkedIn Scraper ---
class LinkedInProvider:
    """Get basic data (name, image, about) from LinkedIn URL using web scraping"""

    def _fetch_data(self, url: str) -> Optional[str]:
        retry_count = 3
        for attempt in range(retry_count):
            user_agent = mimic_bot_headers()
            headers = {"User-Agent": user_agent}
            proxies = {
                "https": "http://brd-customer-hl_6c1f36a6-zone-datacenter_proxy2:1qyqs0lnh5zi@brd.superproxy.io:33335",
                "http": "http://brd-customer-hl_6c1f36a6-zone-datacenter_proxy2:1qyqs0lnh5zi@brd.superproxy.io:33335",
            }

            try:
                response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
                if response.status_code == 200:
                    html = response.text
                    with open("scraped_debug.html", "w", encoding="utf-8") as f:
                        f.write(html)
                    return html
            except Exception as e:
                print(f"Request error for {url}: {e}")
        print(f"❌ Failed to fetch LinkedIn URL after {retry_count} attempts: {url}")
        return None

    def extract_basic_profile_data(self, html: str) -> ProfileData:
        soup = BeautifulSoup(html, 'html.parser')
        data = ProfileData()

        # Extract name
        name_tag = soup.find('h1', class_=lambda x: x and 'top-card-layout__title' in x)
        data.name = name_tag.get_text(strip=True) if name_tag else None

        # Extract canonical LinkedIn URL
        canonical = soup.find('link', rel="canonical")
        data.profile_url = canonical['href'] if canonical and canonical.get('href') else None

        # Extract profile image
        if data.name:
            profile_img = soup.find('img', alt=data.name)
            if profile_img:
                data.profile_image = profile_img.get('data-delayed-url') or profile_img.get('src')

        # Extract "About" section
        about_section = soup.find("section", {"data-section": "summary"})
        if about_section:
            content_div = about_section.find("div", class_="core-section-container__content")
            if content_div:
                data.about = content_div.get_text(separator="\n", strip=True)
        else:
            # Fallback (legacy structure or alternate layout)
            legacy_about = soup.find('section', {'id': 'about'}) or soup.find('div', class_=lambda x: x and 'summary' in x)
            if legacy_about:
                text_blocks = legacy_about.find_all(text=True)
                cleaned_text = ' '.join(t.strip() for t in text_blocks if t.strip())
                data.about = cleaned_text or None

        return data

    def get_profile(self, url: str) -> Tuple[Optional[ProfileData], Optional[Dict]]:
        try:
            html_content = self._fetch_data(url)
            if not html_content:
                return None, None

            profile_data = self.extract_basic_profile_data(html_content)
            return profile_data, {
                "name": profile_data.name,
                "profileUrl": profile_data.profile_url,
                "profileImage": profile_data.profile_image,
                "about": profile_data.about
            }

        except Exception as e:
            print(f"Error extracting profile data: {e}")
            return None, None

# --- Main Processing ---
if __name__ == "__main__":
    input_path = "pipe1_urls.json"
    output_path = "pipe1_scrapped.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    scraper = LinkedInProvider()
    final_output = {}

    for person_name, profiles in data.items():
        print(f"\n🔍 Processing: {person_name}")
        enriched_profiles = []

        for profile in profiles:
            url = profile.get("link")
            if not url:
                continue

            profile_data, raw = scraper.get_profile(url)
            if raw:
                enriched_profiles.append(raw)

        final_output[person_name] = enriched_profiles

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n✅ Done! Output saved to: {output_path}")

from pydantic import BaseModel
from typing import Optional, Dict, List
import json
import re
import difflib
from deepface import DeepFace
from sentence_transformers import SentenceTransformer, util

# Load pretrained model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


class MatchResult(BaseModel):
    name: str
    linkedin_url: Optional[str] = None
    profile_image: Optional[str] = None
    confidence_score: float = 0.0
    match_type: str = "none"


def string_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def is_static_image(url: str) -> bool:
    return "static.licdn.com" in url or "shrinknp" in url  # Add more patterns if needed


def compare_images(input_img_url: str, scraped_img_url: str) -> float:
    try:
        result = DeepFace.verify(
            img1_path=input_img_url,
            img2_path=scraped_img_url,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )
        return round(1 - result["distance"], 2) if result["verified"] else 0.0
    except Exception as e:
        print(f"[Image Match Error] {e}")
        return 0.0


def normalize_image_url(url: str) -> str:
    """Returns a usable image URL (direct Google Drive or regular link)."""
    if not url:
        return ""

    if "drive.google.com" in url:
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
        if match:
            file_id = match.group(1)
            return f"https://drive.google.com/uc?id={file_id}"

    return url  # assume it's already usable


class LinkedInProvider:
    def __init__(self, result_dict: Dict):
        self.result_dict = result_dict

    def get_profile(self, url: str):
        for name, entries in self.result_dict.items():
            for entry in entries:
                if entry.get("profileUrl") == url:
                    return entry, {
                        "name": entry.get("name", ""),
                        "profileUrl": entry.get("profileUrl", ""),
                        "profileImage": entry.get("profileImage", ""),
                        "about": entry.get("about", "")
                    }
        return {}, {}  # Not found


def compute_similarity(input_data: Dict, scraped_data: Dict) -> float:
    scores = []

    if input_data.get("name") and scraped_data.get("name"):
        name_score = string_similarity(input_data["name"], scraped_data["name"])
        scores.append(name_score)

    input_img = normalize_image_url(input_data.get("image", ""))
    scraped_img = scraped_data.get("profileImage", "")
    if input_img and scraped_img and not is_static_image(scraped_img):
        image_score = compare_images(input_img, scraped_img)
        scores.append(image_score)

    intro = input_data.get("intro", "")
    about = scraped_data.get("about", "")
    if intro and about:
        embedding_intro = sbert_model.encode(intro, convert_to_tensor=True)
        embedding_about = sbert_model.encode(about, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding_intro, embedding_about).item()
        scores.append(similarity)

    return round(sum(scores) / len(scores), 2) if scores else 0.0


def evaluate_personas(persona_dataset_path: str, linkedin_results_path: str) -> List[Dict]:
    with open(persona_dataset_path, "r") as f:
        personas = json.load(f)

    with open(linkedin_results_path, "r") as f:
        linkedin_results = json.load(f)

    provider = LinkedInProvider(result_dict=linkedin_results)
    matches = []

    for persona in personas:
        name = persona["name"]
        print(f"🔍 Processing {name}")
        url_dicts = linkedin_results.get(name, [])
        urls = [entry["profileUrl"] for entry in url_dicts if "profileUrl" in entry]

        best_match = None
        best_score = 0.0
        best_type = "none"

        for url in urls:
            print(f"🔍 Checking: {url}")
            profile_data, extracted_data = provider.get_profile(url)
            if not extracted_data:
                print(f"⛔ No data extracted from {url}")
                continue

            score = compute_similarity(input_data=persona, scraped_data=extracted_data)
            print(f"✅ Score: {score} for {url}")

            if score > best_score:
                best_score = score
                scraped_img = extracted_data.get("profileImage", "")
                is_static = is_static_image(scraped_img)
                has_about = bool(extracted_data.get("about"))

                if not is_static and has_about:
                    match_type = "image+text"
                elif not is_static:
                    match_type = "image-only"
                else:
                    match_type = "text-only" if has_about else "name-only"

                best_match = MatchResult(
                    name=name,
                    linkedin_url=extracted_data.get("profileUrl"),
                    profile_image=scraped_img,
                   confidence_score=score,
                    match_type=match_type
                )

        if best_match:
            matches.append(best_match.dict())
        else:
            matches.append(MatchResult(name=name).dict())

    return matches


if __name__ == "__main__":
    results = evaluate_personas(
        persona_dataset_path="pipe_1.json",
        linkedin_results_path="pipe1_scrapped.json"
    )

    with open("pipe1_final.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Processed {len(results)} personas")

import json
import re

with open('pipe_2.json', 'r') as f:
    data = json.load(f)

def clean_name(name):
    # Remove parenthetical parts and add spacing between camelCase words.
    name = re.sub(r"\s*\([^)]*\)", "", name)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.strip()


for person in data:
    person['name'] = clean_name(person['name'])


with open('cleaned_dataset1.json', 'w') as f:
    json.dump(data, f, indent=4)

#-----------------------------------------------------------------------------#
#CREATING QUERY
import json
import re
from keybert import KeyBERT
from flair.data import Sentence
from flair.models import SequenceTagger
from urllib.parse import urlparse

def generate_query(person):
    name = (person.get("name") or "").strip()
    parts = [name]

    return " ".join(part.strip() for part in parts if part)

def process_people(data):
    for person in data:
        person["query"] = generate_query(person)
    return data

if __name__ == "__main__":
    input_file = "cleaned_dataset1.json"
    output_file = "processed_dataset1.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = process_people(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=4)

    print("All entries processed and saved to", output_file)

"""# Linkedin url search"""

import json
import time
from typing import Optional, TypedDict
import requests

class GoogleCustomSearchResponse(TypedDict):
    link: str
    title: Optional[str]
    snippet: Optional[str]

class GoogleCustomSearch:
    def __init__(self):
        self.api_key = "AIzaSyBhoMDnst3_JoK5TziGClgS27bhX0dQNFk"
        self.search_engine_id = "c307652d3fbc44ec8"
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def _search(self, query: str, num=10, start=1) -> Optional[dict]:
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.search_engine_id,
            "num": num,
            "start": start,
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Google Search Failed: {response.status_code} {response.text}")
            return None

    def _parse_results(self, response) -> list[GoogleCustomSearchResponse]:
        if not response or "items" not in response:
            return []
        results = []
        for item in response["items"]:
            link = item.get("link", None)
            if link:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "link": link,
                        "snippet": item.get("snippet", ""),
                    }
                )
        return results

    def search(self, query: str, total_results=15) -> list[GoogleCustomSearchResponse]:
        all_results = []
        remaining = total_results
        start = 1
        while remaining > 0:
            num = min(10, remaining)
            response = self._search(query, num=num, start=start)
            if not response:
                break
            parsed = self._parse_results(response)
            all_results.extend(parsed)
            if len(parsed) < num:
                break  # No more results
            remaining -= num
            start += num
            time.sleep(1)  # Be polite to the API
        return all_results

def main():
    # Load personas from processed.json
    with open('/content/processed_dataset1.json', 'r', encoding='utf-8') as f:
        personas = json.load(f)

    searcher = GoogleCustomSearch()
    results = {}

    for persona in personas:
        name = persona.get("name", "Unknown")
        query = persona.get("query")
        print(f"Searching for: {name} | Query: {query}")
        if not query:
            results[name] = []
            continue
        search_results = searcher.search(f'site:linkedin.com/in {query}', total_results=15)
        results[name] = search_results
        time.sleep(1)  # Be polite to the API

    # Save results to a new JSON file
    with open('linkedin_search_results_dataset1.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("Done! Results saved to linkedin_search_results_dataset1.json")

if __name__ == "__main__":
    main()

"""# Linkedin profile details scrapper"""

from pydantic import BaseModel
from typing import Optional, Dict, Tuple
import requests
import itertools
from bs4 import BeautifulSoup

# --- Data Models ---
class ProfileData(BaseModel):
    name: Optional[str] = None
    profile_url: Optional[str] = None
    profile_image: Optional[str] = None

# --- Rotating User Agents ---
user_agents = [
    "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
    "LinkedInBot/1.0",
    "Twitterbot/1.0",
    "facebookexternalhit/1.1",
    "WhatsApp/2.0",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
]
user_agent_cycle = itertools.cycle(user_agents)

def mimic_bot_headers() -> str:
    """Mimic bot headers by cycling through user agents"""
    return next(user_agent_cycle)

# --- LinkedIn Scraper ---
class LinkedInProvider:
    """Get basic data (name and image) from LinkedIn URL using web scraping"""

    def _fetch_data(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL with proxy and rotating user agents"""
        retry_count = 3

        for _ in range(retry_count):
            user_agent = mimic_bot_headers()

            headers = {
                "User-Agent": user_agent,
            }

            proxies = {
                "https": "http://brd-customer-hl_6c1f36a6-zone-datacenter_proxy2:1qyqs0lnh5zi@brd.superproxy.io:33335",
                "http": "http://brd-customer-hl_6c1f36a6-zone-datacenter_proxy2:1qyqs0lnh5zi@brd.superproxy.io:33335",
            }

            try:
                response = requests.get(
                    url,
                    headers=headers,
                    proxies=proxies,
                    timeout=10
                )

                if response.status_code == 200:
                    return response.text
            except Exception as e:
                print(f"Request error: {e}")

        print(f"Failed to fetch the LinkedIn URL after {retry_count} attempts: {url}")
        return None

    def extract_basic_profile_data(self, html: str) -> ProfileData:
        """Extract just name and profile image from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        data = ProfileData()

        # Extract name
        name_tag = soup.find('h1', class_=lambda x: x and 'top-card-layout__title' in x)
        data.name = name_tag.get_text(strip=True) if name_tag else None

        # Extract profile URL
        canonical = soup.find('link', rel="canonical")
        data.profile_url = canonical['href'] if canonical and canonical.get('href') else None

        # Extract profile image
        if data.name:
            profile_img = soup.find('img', alt=data.name)
            if profile_img:
                data.profile_image = profile_img.get('data-delayed-url') or profile_img.get('src')

        return data

    def get_profile(self, url: str) -> Tuple[Optional[ProfileData], Optional[Dict]]:
        """Get profile data from LinkedIn URL"""
        try:
            html_content = self._fetch_data(url)
            if not html_content:
                return None, None

            profile_data = self.extract_basic_profile_data(html_content)

            # Return both the structured data and raw extracted data
            return profile_data, {
                "name": profile_data.name,
                "profileUrl": profile_data.profile_url,
                "profileImage": profile_data.profile_image
            }

        except Exception as e:
            print(f"Error extracting profile data: {e}")
            return None, None

from pydantic import BaseModel
from typing import Optional, Dict, List
import json
import re
import difflib
from deepface import DeepFace
from sentence_transformers import SentenceTransformer, util
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cosine

# Load pretrained model for semantic similarity
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Data Models ---
class MatchResult(BaseModel):
    name: str
    linkedin_url: Optional[str] = None
    confidence_score: float = 0.0
    match_type: Optional[str] = None

# --- Utility Functions ---
def string_similarity(a: str, b: str) -> float:
    """Calculate string similarity using sequence matcher"""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def compare_images(input_img_url: str, scraped_img_url: str) -> float:
    """Compare faces in two images using DeepFace"""
    try:
        result = DeepFace.verify(
            img1_path=input_img_url,
            img2_path=scraped_img_url,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=False
        )
        return round(1 - result["distance"], 3) if result["verified"] else 0.0
    except Exception as e:
        print(f"[Image Match Error] {e}")
        return 0.0

def convert_drive_link_to_direct(url: str) -> str:
    """Convert Google Drive share links to direct download links"""
    if not url:
        return ""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?id={file_id}"
    return url

# --- Confidence Score Calculation ---
def compute_similarity(input_data: Dict, scraped_data: Dict) -> tuple:
    """Compute similarity score between input data and scraped profile
    Returns a tuple of (final_score, match_type, name_score, image_score)
    """
    scores = {}
    name_score = 0.0
    image_score = 0.0

    has_input_image = "image" in input_data and input_data["image"]
    has_scraped_image = "profileImage" in scraped_data and scraped_data["profileImage"]

    # Compare names
    if input_data.get("name") and scraped_data.get("name"):
        name_score = string_similarity(input_data["name"], scraped_data["name"])
        scores["name"] = name_score

    # Compare images
    if has_input_image and has_scraped_image:
        input_img = convert_drive_link_to_direct(input_data["image"])
        scraped_img = convert_drive_link_to_direct(scraped_data["profileImage"])

        try:
            image_score = compare_images(input_img, scraped_img)
            scores["image"] = image_score
        except Exception as e:
            print(f"[Image Comparison Error] {e}")

    # Determine match type and final score
    if "name" in scores and "image" in scores:
        final_score = (scores["name"] * 0.3) + (scores["image"] * 0.7)
        match_type = "name_and_image"
    elif "name" in scores:
        final_score = min(scores["name"], 0.5)
        match_type = "name_only"
    elif "image" in scores:
        final_score = min(scores["image"], 0.6)
        match_type = "image_only"
    else:
        final_score = 0.0
        match_type = "no_match"

    return round(final_score, 3), match_type, round(name_score, 3), round(image_score, 3)

# --- Main Process ---
def evaluate_personas(persona_dataset_path: str, linkedin_results_path: str) -> List[Dict]:
    """Process all personas and find best matches"""
    # Load data
    with open(persona_dataset_path, "r") as f:
        personas = json.load(f)

    with open(linkedin_results_path, "r") as f:
        linkedin_results = json.load(f)

    matches = []

    for persona in personas:
        name = persona["name"]
        print(f"🔍 Processing {name}")

        # Get LinkedIn URLs from search results
        url_dicts = linkedin_results.get(name, [])
        urls = [entry["link"] for entry in url_dicts]

        if not urls:
            matches.append(MatchResult(
                name=name,
                confidence_score=0.0,
                name_similarity=0.0,
                image_similarity=0.0,
                match_type="no_results"
            ).dict())
            continue

        best_match = None
        best_score = 0
        match_type = "no_match"

        provider = LinkedInProvider()

        first_url = urls[0]
        first_profile_data = None
        first_extracted_data = None

        for url in urls:
            print(f"🔍 Scraping: {url}")
            try:
                profile_data, extracted_data = provider.get_profile(url)

                if url == first_url:
                    first_profile_data = profile_data
                    first_extracted_data = extracted_data

                if not profile_data or not extracted_data:
                    continue

                score, current_match_type, name_sim, image_sim = compute_similarity(
                    input_data=persona,
                    scraped_data=extracted_data
                )

                if score > best_score:
                    best_score = score
                    match_type = current_match_type
                    best_name_score = name_sim
                    best_image_score = image_sim

                    best_match = MatchResult(
                        name=name,
                        linkedin_url=extracted_data.get("profileUrl", url),
                        confidence_score=score,
                        match_type=match_type
                    )

            except Exception as e:
                print(f"[Scraping Error] {e}")
                continue

        has_input_image = "image" in persona and persona["image"]

        if not best_match:
            if first_extracted_data:
                name_score = string_similarity(name, first_extracted_data.get("name", ""))
                if not has_input_image:
                    adjusted_score = min(name_score, 0.4)
                    fallback_type = "name_only_fallback"
                else:
                    adjusted_score = min(name_score * 0.4, 0.4)
                    fallback_type = "partial_match_fallback"

                best_match = MatchResult(
                    name=name,
                    linkedin_url=first_extracted_data.get("profileUrl", first_url),
                    confidence_score=round(adjusted_score, 2),
                    match_type=fallback_type
                )
            else:
                best_match = MatchResult(
                    name=name,
                    linkedin_url=first_url,
                    confidence_score=0.1,
                    match_type="url_only_fallback"
                )

        matches.append(best_match.dict())

    return matches


if __name__ == "__main__":
    results = evaluate_personas(
        persona_dataset_path="pipe_2.json",
        linkedin_results_path="linkedin_search_results_dataset1.json"
    )

    # Save results
    with open("pipe2_final.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Processed {len(results)} personas and saved results.")

import json

# Load the dataset
with open("pipe_3.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Helper function to check if a value is empty
def is_empty(val):
    return (
        val is None or
        (isinstance(val, str) and val.strip() == "") or
        (isinstance(val, list) and len(val) == 0)
    )

# Clean & flexible filtering logic
filtered_profiles = [
    profile for profile in data
    if is_empty(profile.get("intro")) and (
        not is_empty(profile.get("company_industry")) or
        not is_empty(profile.get("company_size"))
    )
]

# Save to pipe3.json
with open("pipe3_processed.json", "w", encoding="utf-8") as outfile:
    json.dump(filtered_profiles, outfile, indent=2, ensure_ascii=False)

print(f"Saved {len(filtered_profiles)} profiles to pipe3.json")

import json

# Load the dataset
with open("pipe3_processed.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Update the 'query' field
for profile in data:
    name = profile.get("name", "")
    industry = profile.get("company_industry")

    # Check if industry is valid and non-empty
    if industry and isinstance(industry, str) and industry.strip():
        updated_query = f"{name} {industry.strip().lower()}"
    else:
        updated_query = name  # If no industry, keep only the name

    profile["query"] = updated_query

# Save the updated dataset (optional: you can overwrite or save as new)
with open("pipe3_query.json", "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, indent=2, ensure_ascii=False)

print("Updated 'query' field based on name and company_industry.")

import re
import json

def extract_range(size_str):
    if not size_str:
        return None

    size_str = size_str.lower()
    numbers = list(map(int, re.findall(r'\d+', size_str)))

    if len(numbers) == 2:
        return f"{numbers[0]}-{numbers[1]}"
    elif len(numbers) == 1:
        if any(kw in size_str for kw in ['fewer', 'less', 'under']):
            return f"0-{numbers[0]}"
        else:
            return f"{numbers[0]}-{numbers[0]}"
    else:
        return None

# Load your JSON (replace with actual file or string load)
with open('pipe3_query.json', 'r') as f:
    data = json.load(f)

# Update company_size
for entry in data:
    entry['company_size'] = extract_range(entry.get('company_size'))

# Save the updated JSON
with open('processed_3.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Updated JSON saved as 'processed.json'")

import json
import time
import requests
from typing import Optional, TypedDict, List

class GoogleCustomSearchResponse(TypedDict):
    link: str
    title: Optional[str]
    snippet: Optional[str]

class GoogleCustomSearch:
    def __init__(self):  # Fixed: should be __init__
        self.api_key = "AIzaSyBhoMDnst3_JoK5TziGClgS27bhX0dQNFk"
        self.search_engine_id = "c307652d3fbc44ec8"
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def _search(self, query: str, num=10, start=1) -> Optional[dict]:
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.search_engine_id,
            "num": num,
            "start": start,
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Google Search Failed: {response.status_code} {response.text}")
            return None

    def _parse_results(self, response) -> List[GoogleCustomSearchResponse]:
        if not response or "items" not in response:
            return []
        results = []
        for item in response["items"]:
            link = item.get("link", None)
            if link:
                results.append({
                    "title": item.get("title", ""),
                    "link": link,
                    "snippet": item.get("snippet", ""),
                })
        return results

    def search(self, query: str, num=5, start=1) -> List[GoogleCustomSearchResponse]:
        response = self._search(query, num, start)
        if response:
            return self._parse_results(response)
        return []

def main():
    # Load personas from processed.json
    with open('processed_3.json', 'r', encoding='utf-8') as f:
        personas = json.load(f)

    searcher = GoogleCustomSearch()
    results = {}

    for persona in personas:
        name = persona.get("name", "Unknown")
        query = persona.get("query")
        print(f"Searching for: {name} | Query: {query}")
        if not query:
            results[name] = []
            continue
        search_results = searcher.search(f'site:linkedin.com/in {query}', num=10)
        results[name] = search_results
        time.sleep(1)  # Be polite to the API

    # Save results to a new JSON file
    with open('pipe3_urls.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("Done! Results saved to pipe3_urls.json")

if __name__ == "__main__":  # Fixed: should be __name__
    main()

import json
import re
import itertools
from typing_extensions import Optional, TypedDict

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

# List of user agents used in rotation
user_agents = [
    "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
    "LinkedInBot/1.0",
    "Twitterbot/1.0",
    "facebookexternalhit/1.1",
    "WhatsApp/2.0",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
]

user_agent_cycle = itertools.cycle(user_agents)


def mimic_bot_headers() -> str:
    """
    Mimic bot headers
    """
    return next(user_agent_cycle)


def get_first_last_name(name: str) -> tuple[str, Optional[str]]:
    """
    Extracts first and last name from a full name.
    """
    name_parts = name.split(" ")
    first_name = name_parts[0]
    last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else None
    return first_name, last_name


# TypedDict and Pydantic models for profile data
class Workspace(TypedDict):
    name: str
    url: Optional[str]


class LinkedinPersonProfile(BaseModel):
    first_name: Optional[str]
    last_name: Optional[str]
    linkedin: Optional[str]
    workspaces: Optional[list[Workspace]]


class LinkedinCompanyProfile(BaseModel):
    name: Optional[str]
    website: Optional[str]
    description: Optional[str]
    address: Optional[str]
    number_of_employees: Optional[int]


class LinkedInProvider:
    """
    Get data from a LinkedIn URL using web scraping.
    """

    def _fetch_data(self, url: str) -> Optional[str]:
        retry_count = 3
        for _ in range(retry_count):
            user_agent = mimic_bot_headers()
            headers = {"User-Agent": user_agent}
            proxies = {
                "https": "http://brd-customer-hl_6c1f36a6-zone-datacenter_proxy2:1qyqs0lnh5zi@brd.superproxy.io:33335",
                "http": "http://brd-customer-hl_6c1f36a6-zone-datacenter_proxy2:1qyqs0lnh5zi@brd.superproxy.io:33335",
            }
            response = requests.get(url, headers=headers, proxies=proxies)
            if response.status_code == 200:
                return response.text
        print(f"Failed to fetch the LinkedIn URL: {url}")
        return None

    def _json_ld_data(self, html_content: str) -> Optional[dict]:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            script_tag = soup.find("script", {"type": "application/ld+json"})
            json_ld_data = json.loads(script_tag.string) if script_tag else {}
            return json_ld_data
        except Exception as e:
            print(f"Error extracting JSON-LD data: {e}")
            return None

    def person_profile(self, url: str) -> Optional[tuple[LinkedinPersonProfile, dict]]:
        """
        Extracts the profile details of a person.
        Returns a tuple containing the Pydantic model and additional scraped data.
        """
        try:
            html_content = self._fetch_data(url)
            if not html_content:
                return None

            # Use a helper function to scrape additional profile details
            data_obj = extract_profile_data(html_content)
            json_ld_data = self._json_ld_data(html_content)

            if json_ld_data:
                # If JSON-LD data represents a ProfilePage, use "mainEntity", otherwise iterate the graph
                if json_ld_data.get("@type") == "ProfilePage":
                    person_data = json_ld_data["mainEntity"]
                else:
                    person_data = next(
                        (
                            item
                            for item in json_ld_data.get("@graph", [])
                            if item.get("@type") == "Person"
                        ),
                        {},
                    )

                name = person_data.get("name")
                workplaces = [
                    {"name": org.get("name"), "url": org.get("url")}
                    for org in person_data.get("worksFor", [])
                    if "name" in org
                ]
                first_name, last_name = (get_first_last_name(name) if name else (None, None))

                profile = LinkedinPersonProfile(
                    first_name=first_name,
                    last_name=last_name,
                    linkedin=url,
                    workspaces=workplaces,
                )
                return profile, data_obj

        except Exception as e:
            print(f"Error in extracting person profile: {e}")
            return None

    def company_profile(self, username: str) -> Optional[LinkedinCompanyProfile]:
        """
        Extracts the profile details of a company.
        """
        try:
            url = f"https://www.linkedin.com/company/{username}"
            html_content = self._fetch_data(url)
            if not html_content:
                return None

            json_ld_data = self._json_ld_data(html_content)
            if json_ld_data:
                if json_ld_data.get("@type") == "ProfilePage":
                    organization_data = json_ld_data["mainEntity"]
                else:
                    organization_data = next(
                        (
                            item
                            for item in json_ld_data.get("@graph", [])
                            if item.get("@type") == "Organization"
                        ),
                        {},
                    )

                name = organization_data.get("name")
                website = organization_data.get("sameAs")
                description = organization_data.get("description")
                number_of_employees = organization_data.get("numberOfEmployees", {}).get("value")

                # Format the address
                address_dict = organization_data.get("address", {})
                address_parts = [
                    address_dict.get("streetAddress"),
                    address_dict.get("addressLocality"),
                    address_dict.get("addressRegion"),
                    address_dict.get("postalCode"),
                    address_dict.get("addressCountry"),
                ]
                address = ", ".join(filter(None, address_parts))
                return LinkedinCompanyProfile(
                    name=name,
                    website=website,
                    description=description,
                    address=address,
                    number_of_employees=number_of_employees,
                )
        except Exception as e:
            print(f"Error in extracting company profile: {e}")
            return None


def extract_profile_data(html: str) -> dict:
    """
    Scrapes additional profile data from HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    data = {}

    name_tag = soup.find("h1", class_=lambda x: x and "top-card-layout__title" in x)
    data["name"] = name_tag.get_text(strip=True) if name_tag else None

    canonical = soup.find("link", rel="canonical")
    data["profileUrl"] = canonical["href"] if canonical and canonical.get("href") else None

    profile_img = soup.find("img", alt=data["name"])
    if profile_img:
        data["profileImage"] = profile_img.get("data-delayed-url") or profile_img.get("src")
    else:
        data["profileImage"] = None

    about_section = soup.find("section", class_=lambda x: x and "summary" in x)
    if about_section:
        p_about = about_section.find("p")
        data["about"] = p_about.get_text(strip=True) if p_about else None
    else:
        data["about"] = None

    # Extract experience items
    exp_items = []
    exp_section = soup.find("section", attrs={"data-section": "experience"})
    if exp_section:
        for li in exp_section.find_all("li", class_=lambda x: x and "experience-item" in x):
            company = None
            location = None
            company_tag = li.find("span", class_=lambda x: x and "experience-item__subtitle" in x)
            if company_tag:
                company = company_tag.get_text(strip=True)
            location_tag = li.find("p", class_=lambda x: x and "experience-item__meta-item" in x)
            if location_tag:
                location = location_tag.get_text(strip=True)
            if company or location:
                exp_items.append({"company": company, "location": location})
    data["experience"] = exp_items

    # Extract education items
    edu_items = []
    edu_section = soup.find("section", attrs={"data-section": "educationsDetails"})
    if edu_section:
        for li in edu_section.find_all("li", class_=lambda x: x and "education__list-item" in x):
            institution = None
            period = None
            description = None
            inst_link = li.find("a", href=lambda href: href and "school" in href)
            if inst_link:
                institution = inst_link.get_text(strip=True)
            period_tag = li.find("span", class_=lambda x: x and "date-range" in x)
            if period_tag:
                period = period_tag.get_text(strip=True)
            desc_div = li.find("div", class_=lambda x: x and "show-more-less-text" in x)
            if desc_div:
                description = desc_div.get_text(" ", strip=True)
            edu_items.append({"institution": institution, "period": period, "description": description})
    data["education"] = edu_items

    # Extract languages
    languages = []
    lang_section = soup.find("section", class_=lambda x: x and "languages" in x)
    if lang_section:
        for li in lang_section.find_all("li"):
            language = None
            proficiency = None
            lang_name_tag = li.find("h3")
            prof_tag = li.find("h4")
            if lang_name_tag:
                language = lang_name_tag.get_text(strip=True)
            if prof_tag:
                proficiency = prof_tag.get_text(strip=True)
            if language:
                languages.append({"language": language, "proficiency": proficiency})
    data["languages"] = languages

    # Extract recommendations count if available
    recommendations_received = None
    rec_section = soup.find("section", class_=lambda x: x and "recommendations" in x)
    if rec_section:
        rec_text = rec_section.get_text(" ", strip=True)
        m = re.search(r"(\d+)\s+people\s+have\s+recommended", rec_text)
        if m:
            recommendations_received = int(m.group(1))
    data["recommendationsReceived"] = recommendations_received

    return data


def process_urls_from_file(input_file: str, output_file: str):
    """
    Reads a JSON file with LinkedIn URLs, processes each URL to extract profile data,
    and writes the results to an output JSON file.
    """
    # Load the JSON file containing URLs.
    with open(input_file, "r", encoding="utf-8") as f:
        url_data = json.load(f)

    profiles = {}
    provider = LinkedInProvider()

    for person_name, entries in url_data.items():
        profiles[person_name] = []
        for entry in entries:
            link = entry.get("link")
            if link:
                print(f"Processing URL: {link}")
                profile_result = provider.person_profile(link)
                if profile_result:
                    profile_obj, complete_data = profile_result
                    profiles[person_name].append({
                        "profile": profile_obj.dict(),
                        "scraped_data": complete_data,
                        "source_title": entry.get("title"),
                        "source_snippet": entry.get("snippet")
                    })
                else:
                    profiles[person_name].append({
                        "error": f"Could not retrieve data for {link}",
                        "source_title": entry.get("title")
                    })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print(f"Processed profiles have been saved to {output_file}")


if __name__ == "__main__":
    input_json_file = "pipe3_urls.json"
    output_json_file = "pipe3_scrapped.json"
    process_urls_from_file(input_json_file, output_json_file)

import json

with open("pipe3_scrapped.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output = {}

for main_name, profiles in data.items():
    new_profiles = []
    for profile_entry in profiles:
        profile = profile_entry.get("profile", {})
        scraped_data = profile_entry.get("scraped_data", {})

        name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
        linkedin = profile.get("linkedin")
        profile_image = scraped_data.get("profileImage")

        workspaces = profile.get("workspaces", [])
        companies = [
            {"name": ws["name"], "url": ws["url"]}
            for ws in workspaces if ws.get("url")
        ]

        new_profiles.append({
            "name": name,
            "linkedin": linkedin,
            "profile_image": profile_image,
            "companies": companies
        })

    output[main_name] = new_profiles

with open("pipe3_companies.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("Done! Saved to pipe3_companies.json")

import json
import re
import itertools
from typing_extensions import Optional
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

# List of user agents used in rotation
user_agents = [
    "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
    "LinkedInBot/1.0",
    "Twitterbot/1.0",
    "facebookexternalhit/1.1",
    "WhatsApp/2.0",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
]

user_agent_cycle = itertools.cycle(user_agents)


def mimic_bot_headers() -> str:
    """Mimic bot headers"""
    return next(user_agent_cycle)


class LinkedinCompanyProfile(BaseModel):
    name: Optional[str]
    website: Optional[str]
    description: Optional[str]
    address: Optional[str]
    number_of_employees: Optional[int]


class LinkedInProvider:
    """
    Get data from a LinkedIn company URL using web scraping.
    """

    def _fetch_data(self, url: str) -> Optional[str]:
        retry_count = 3
        for _ in range(retry_count):
            user_agent = mimic_bot_headers()
            headers = {"User-Agent": user_agent}
            proxies = {
                "https": "http://brd-customer-hl_6c1f36a6-zone-datacenter_proxy2:1qyqs0lnh5zi@brd.superproxy.io:33335",
                "http": "http://brd-customer-hl_6c1f36a6-zone-datacenter_proxy2:1qyqs0lnh5zi@brd.superproxy.io:33335",
            }
            response = requests.get(url, headers=headers, proxies=proxies)
            if response.status_code == 200:
                return response.text
        print(f"Failed to fetch the LinkedIn URL: {url}")
        return None

    def _json_ld_data(self, html_content: str) -> Optional[dict]:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            script_tag = soup.find("script", {"type": "application/ld+json"})
            json_ld_data = json.loads(script_tag.string) if script_tag else {}
            return json_ld_data
        except Exception as e:
            print(f"Error extracting JSON-LD data: {e}")
            return None

    def company_profile(self, username: str) -> Optional[LinkedinCompanyProfile]:
        """
        Extracts the profile details of a company.
        """
        try:
            url = f"https://www.linkedin.com/company/{username}"
            html_content = self._fetch_data(url)
            if not html_content:
                return None

            json_ld_data = self._json_ld_data(html_content)
            if json_ld_data:
                if json_ld_data.get("@type") == "ProfilePage":
                    organization_data = json_ld_data["mainEntity"]
                else:
                    organization_data = next(
                        (
                            item
                            for item in json_ld_data.get("@graph", [])
                            if item.get("@type") == "Organization"
                        ),
                        {},
                    )

                name = organization_data.get("name")
                website = organization_data.get("sameAs")
                description = organization_data.get("description")
                number_of_employees = organization_data.get("numberOfEmployees", {}).get("value")

                address_dict = organization_data.get("address", {})
                address_parts = [
                    address_dict.get("streetAddress"),
                    address_dict.get("addressLocality"),
                    address_dict.get("addressRegion"),
                    address_dict.get("postalCode"),
                    address_dict.get("addressCountry"),
                ]
                address = ", ".join(filter(None, address_parts))

                return LinkedinCompanyProfile(
                    name=name,
                    website=website,
                    description=description,
                    address=address,
                    number_of_employees=number_of_employees,
                )
        except Exception as e:
            print(f"Error in extracting company profile: {e}")
            return None


def process_company_urls_from_file(input_file: str, output_file: str):
    """
    Reads a JSON file with company URLs, processes each URL to extract company profile data,
    and writes the results to an output JSON file.
    Includes the person's name and LinkedIn profile URL in each record.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    provider = LinkedInProvider()
    output_data = {}

    for person_name, entries in input_data.items():
        output_data[person_name] = []

        for entry in entries:
            user_name = entry.get("name", "")
            user_profile_url = entry.get("url", "")
            companies = entry.get("companies", [])

            for company in companies:
                company_url = company.get("url")
                if company_url and "linkedin.com/company/" in company_url:
                    match = re.search(r"linkedin\.com/company/([^/?#]+)", company_url)
                    if match:
                        company_username = match.group(1)
                        print(f"Processing company: {company_username}")
                        profile = provider.company_profile(company_username)

                        base_data = {
                            "name": user_name,
                            "linkedin_profile_url": user_profile_url,
                            "company_url": company_url
                        }

                        if profile:
                          output_data[person_name].append({
    "company_name": profile.name,
    "company_url": company_url,
    "person_name": person_name,
    "person_profile_url": entry.get("linkedin", "Unknown"),
    "person_image_url": entry.get("profile_image", "Unknown"),
    "company_profile": profile.dict()
})


                        else:
                            output_data[person_name].append({
                                **base_data,
                                "error": f"Failed to extract company profile for {company_username}"
                            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Company profiles saved to {output_file}")


if __name__ == "__main__":
    input_json_file = "pipe3_companies.json"  # Your input file
    output_json_file = "pipe3_company_profiles_output.json"
    process_company_urls_from_file(input_json_file, output_json_file)

import json

# Helper to convert a string like "0-50" into a numeric range
def parse_range(size_str):
    if not size_str or "-" not in size_str:
        return None, None
    try:
        low, high = size_str.split("-")
        return int(low), int(high)
    except ValueError:
        return None, None

# Load profile and company data
with open("processed_3.json", "r") as f:
    profiles = json.load(f)

with open("pipe3_company_profiles_output.json", "r") as f:
    matches = json.load(f)

# Validate matches
results = []

for profile in profiles:
    name = profile.get("name")
    size_range = profile.get("company_size")

    low, high = parse_range(size_range)

    if name not in matches:
        continue  # skip if no companies found

    person_matches = matches[name]
    for match in person_matches:
        company = match.get("company_name")
        company_profile = match.get("company_profile", {})
        num_employees = company_profile.get("number_of_employees")
        person_url = match.get("person_profile_url")

        # Check if the range is missing
        if low is None or high is None:
            results.append({
                "person_name": name,
                "person_profile_url": person_url,
                "company_name": company,
                "num_employees": num_employees,
                "expected_range": size_range,
                "match": False
            })
            continue

        # Compare number of employees against expected range
        if isinstance(num_employees, int):
            in_range = low <= num_employees <= high
            results.append({
                "person_name": name,
                "person_profile_url": person_url,
                "company_name": company,
                "num_employees": num_employees,
                "expected_range": size_range,
                "match": in_range
            })

# Output results
with open("pipe3_company_size_validation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Validation complete. Results saved to pipe3_company_size_validation_results.json.")

import json
from collections import defaultdict

# Load data
with open("pipe3_company_size_validation_results.json", "r") as f:
    validation_data = json.load(f)

with open("pipe3_urls.json", "r") as f:
    linkedin_search = json.load(f)

# Helper to parse range like "0-50"
def parse_range(size_str):
    try:
        low, high = map(int, size_str.split("-"))
        return low, high
    except:
        return None, None

# Group entries by person
grouped = defaultdict(list)
for entry in validation_data:
    grouped[entry["person_name"]].append(entry)

final_output = []

for name, records in grouped.items():
    # Step 1: Filter records where match is True
    true_matches = [r for r in records if r["match"]]

    if true_matches:
        # Step 2: Pick best match based on closest to expected range average
        low, high = parse_range(true_matches[0]["expected_range"])
        mid = (low + high) / 2 if low is not None and high is not None else 0

        best_match = min(true_matches, key=lambda r: abs(r["num_employees"] - mid))

        # Step 3: Calculate similarity using improved formula
        if mid > 0:
            deviation = abs((mid - best_match["num_employees"]) / mid) * 100
            similarity = max(0, 100 - deviation)
        else:
            similarity = 0

        final_output.append({
            "person_name": name,
            "match": True,
            "person_profile_url": best_match["person_profile_url"],
            "expected_range": best_match["expected_range"],
            "similarity": round(similarity, 2)
        })
    else:
        # Step 4: No match — determine similarity if range exists, else fallback
        expected_range = records[0]["expected_range"]

        if expected_range:
            low, high = parse_range(expected_range)
            mid = (low + high) / 2 if low is not None and high is not None else 0

            best_guess = min(
                records,
                key=lambda r: abs(r["num_employees"] - mid) if mid else float('inf')
            )

            if mid > 0:
                deviation = abs((mid - best_guess["num_employees"]) / mid) * 100
                similarity = max(0, 100 - deviation)
            else:
                similarity = 0

            profile_url = best_guess["person_profile_url"]
        else:
            similarity = 0
            profile_url = linkedin_search.get(name, [{}])[0].get("link", None)

        final_output.append({
            "person_name": name,
            "match": False,
            "person_profile_url": profile_url,
            "expected_range": expected_range,
            "similarity": round(similarity, 2)
        })

# Save result
with open("pipe3_final_companysize_validation_output.json", "w") as f:
    json.dump(final_output, f, indent=2)

print("Final output saved to pipe3_final_companysize_validation_output.json")

import json
import pandas as pd

# Load JSONs
with open("pipe3_company_profiles_output.json", "r", encoding="utf-8") as f:
    linkedin_data = json.load(f)

with open("processed_3.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Build train_images and train_profile_urls from linkedin_data
train_images = {}
train_profile_urls = {}

for person_name, profiles in linkedin_data.items():
    pics = []
    profile_links = []
    for p in profiles:
        img_url = p.get("person_image_url")
        profile_url = p.get("person_profile_url")
        if isinstance(img_url, str) and img_url.startswith("https://media.licdn.com/dms/image/" or "https://www.capterra.com/"):
            pics.append(img_url)
            profile_links.append(profile_url)
    if pics:
        train_images[person_name] = pics
        train_profile_urls[person_name] = profile_links

# Build test_images
test_images = {}
for entry in test_data:
    name = entry.get("name")
    url = entry.get("image") or entry.get("image_url") or entry.get("person_image_url")
    if name and isinstance(url, str):
        test_images[name] = [url]

# Find intersection
common = set(train_images) & set(test_images)

# Assemble DataFrame
rows = []
for name in sorted(common):
    rows.append({
        "name": name,
        "train_imgs": train_images[name],
        "train_profile_urls": train_profile_urls[name],
        "actual_imgs": test_images[name]
    })
df = pd.DataFrame(rows)


import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
import re
from urllib.parse import urlparse, parse_qs, unquote

os.makedirs("train_images", exist_ok=True)
os.makedirs("actual_images", exist_ok=True)

# Converts Google Drive links to direct download
def convert_drive_to_direct(url):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

# Extracts real image URL if it's wrapped in a _next/image proxy
def extract_direct_image_url(url):
    if "_next/image" in url and "url=" in url:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        if 'url' in query_params:
            return unquote(query_params['url'][0])
    return url

# Downloads and saves images
def download_and_save_images(image_urls, folder, name_prefix):
    for i, url in enumerate(image_urls):
        try:
            direct_url = extract_direct_image_url(convert_drive_to_direct(url))
            response = requests.get(direct_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(os.path.join(folder, f"{name_prefix}_{i}.jpg"), "JPEG")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# Loop through each row and download train/actual images
for idx, row in df.iterrows():
    name = row['name'].replace(" ", "_")
    if isinstance(row['train_imgs'], list):
        download_and_save_images(row['train_imgs'], "train_images", f"{name}_train")
    if isinstance(row['actual_imgs'], list):
        download_and_save_images(row['actual_imgs'], "actual_images", f"{name}_actual")

import os
import json
import numpy as np
import pandas as pd
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity



def get_embedding(img_path, model_name='VGG-Face'):
    try:
        embedding_obj = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)[0]
        return embedding_obj['embedding']
    except Exception as e:
        print(f"Error on {img_path}: {e}")
        return None

def cosine_sim(emb1, emb2):
    return cosine_similarity([emb1], [emb2])[0][0]

names = [f.split('_actual_')[0] for f in os.listdir("actual_images") if f.endswith('.jpg')]
results = {}

for name in names:
    actual_path = f"actual_images/{name}_actual_0.jpg"
    if not os.path.exists(actual_path):
        continue
    actual_emb = get_embedding(actual_path)
    if actual_emb is None:
        continue

    max_score = -1
    best_train_path = None

    i = 0
    while True:
        train_path = f"train_images/{name}_train_{i}.jpg"
        if not os.path.exists(train_path):
            break
        train_emb = get_embedding(train_path)
        if train_emb is not None:
            score = cosine_sim(train_emb, actual_emb)
            if score > max_score:
                max_score = score
                best_train_path = train_path
                best_index = i
        i += 1

    if max_score >= 0.20:
        # Extract clean name and index
        filename = os.path.basename(best_train_path)
        parts = filename.split('_train_')
        person_clean = parts[0].replace('_', ' ')
        index = int(parts[1].split('.')[0])

        # Fetch profile URL
        try:
            urls_array = np.array(df[df['name'] == person_clean]['train_profile_urls'])[0]
            person_url = urls_array[index]
        except Exception as e:
            print(f"URL fetch failed for {person_clean} at index {index}: {e}")
            person_url = None

        results[name] = {
            "similarity": round(max_score, 4),
            "person_profile_url": person_url
        }
    else:
        results[name] = "No similarity found"

# Save result to JSON
with open("pipe3_face_similarity_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to pipe3_face_similarity_results.json")

import json

# Load JSON data
with open("pipe3_face_similarity_results.json") as f:
    face_data = json.load(f)

with open("pipe3_final_companysize_validation_output.json") as f:
    linkedin_data = json.load(f)

# Process each person
for person in linkedin_data:
    name_key = person["person_name"].replace(" ", "_")
    face_match = face_data.get(name_key)

    if face_match and "person_profile_url" in face_match:
        # Override URL + use similarity from face JSON
        person["person_profile_url"] = face_match["person_profile_url"]
        person["confidence_score"] = float(face_match.get("similarity"))/100
    else:
        # Keep original URL + use existing similarity from match JSON as confidence_score
        person["confidence_score"] = float(person.get("similarity"))/100

    # Remove 'match' and old 'similarity' keys if they exist
    person.pop("match", None)
    person.pop("similarity", None)

# Save result
with open("pipe3_final.json", "w") as f:
    json.dump(linkedin_data, f, indent=4)

results=[]

with open("pipe1_final.json", "r") as f:
    data1 = json.load(f)

for item in data1:
    results.append(item)

with open("pipe2_final.json", "r") as f:
    data2 = json.load(f)

for item in data2:
    results.append(item)

with open("pipe3_final.json","r") as f:
  results.append(json.load(f))

with open("results.json","w") as f:
  json.dump(results,f,indent=4)

print(json.dumps(results, indent=4))

