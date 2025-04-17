# LinkedIn Profile Matching System

## Features

- **Three specialized pipelines** for different input data scenarios
- **Hybrid matching** using name similarity, facial recognition, and text embeddings
- **Google Custom Search API** for profile discovery
- **Rotating proxy support** for reliable scraping
- **JSON-based data handling** for easy integration

# Setup the Project

Follow the steps below to set up the project environment:

## 1. Clone the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/VamshiNarmety/Linkedin_profile_match
```
## 2. Create a Virtual Environment
Navigate to the project directory and create a virtual environment by running the following command in the terminal:
```bash python3 -m venv venv```
## 3. Activate the Virtual Environment
## 4. Install Python 3.10.x

## 5. Install Dependencies from `requirements.txt`

With the virtual environment activated, install the required dependencies by running the following command:

```bash pip install -r requirements.txt```


## Configuration
Fill in the API details and Proxy credentials in .env _____________.

---

google_api:
key: "your_google_cse_key"
engine_id: "your_search_engine_id"

scraping:
proxies:
- "http://user:pass@proxy1:port"
- "http://user:pass@proxy2:port"
user_agents: "user_agents.txt"


---




## Environment Setup
Required API Keys (set as environment variables)
GOOGLE_API_KEY = "your_google_api_key"
SEARCH_ENGINE_ID = "your_search_engine_id"
PROXY_CREDENTIALS = "user:pass" # For scraping


## Example Input and Output formats:


---





---



## Pipeline Architecture

### Pipeline 1 (Name+Intro based)
1.Generates a query with Name and two unique keywords from their intro.
2.Uses Google Custom Search API to search for LinkedIn URLs.
3.Extracts Name, Profile image, About section, Canonical LinkedIn URL by scraping.
4.Name similarity using sequence matching and Image similarity using DeepFace verification and Intro/About text similarity using Sentence Transformers.
5.Confidence scoring

### Pipeline 2 (Name+Image based)

1.Generates a query with only name
2.Uses Google Custom Search API to search for LinkedIn URLs.
3.Extract Name,Profile image by scraping
3.Name similarity using sequence matching and Image similarity using DeepFace verification.
4.Confidence scoring

### Pipeline 3 (Name+Company_Industry/Company_size based)

1.Name and Company_Industry/Company_size search
2.Uses Google Custom Search API to search for LinkedIn URLs.
3.Extracts Name,Profile image, Work History, Education, etc. by scraping.
4.Processes the scraped profiles to extract associated companies(with LinkedIn URLs) for each person.
5.Company profile scraping.
6.Company size validation.
7.Best match selection and Confidence scoring.
