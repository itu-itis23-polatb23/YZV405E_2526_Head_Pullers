import os
import json
import random
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
}

URLS = {
    "French": "https://en.wiktionary.org/wiki/Category:French_idioms",
    "Georgian": "https://en.wiktionary.org/wiki/Category:Georgian_idioms",
    "Greek": "https://en.wiktionary.org/wiki/Category:Greek_idioms",
    "Japanese": "https://en.wiktionary.org/wiki/Category:Japanese_idioms",
    "Hebrew": "https://en.wiktionary.org/wiki/Category:Hebrew_idioms",
    "Latvian": "https://en.wiktionary.org/wiki/Category:Latvian_idioms",
    "Persian": "https://en.wiktionary.org/wiki/Category:Persian_idioms",
    "Polish": "https://en.wiktionary.org/wiki/Category:Polish_idioms",
    "Portuguese": "https://en.wiktionary.org/wiki/Category:Portuguese_idioms",
    "Romanian": "https://en.wiktionary.org/wiki/Category:Romanian_idioms",
    "Serbian(Serbo-Croatian)": "https://en.wiktionary.org/wiki/Category:Serbo-Croatian_idioms",
    "Slovene": "https://en.wiktionary.org/wiki/Category:Slovene_idioms",
    "Swedish": "https://en.wiktionary.org/wiki/Category:Swedish_idioms",
    "Ukrainian": "https://en.wiktionary.org/wiki/Category:Ukrainian_idioms"
}

def get_idiom_links(category_url):
    links = []
    current_url = category_url
    
    while current_url:
        response = requests.get(current_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Idioms are usually listed in the mw-pages div
        pages_div = soup.find('div', id='mw-pages')
        if not pages_div:
            break
            
        for a_tag in pages_div.find_all('a'):
            href = a_tag.get('href')
            title = a_tag.get('title')
            # Exclude subcategories and functional pages
            if href and title and not title.startswith('Category:'):
                links.append((title, 'https://en.wiktionary.org' + href))
                
        # Handle pagination to next contents page
        next_link = None
        for a_tag in pages_div.find_all('a'):
            if a_tag.get_text() == 'next page':
                next_link = 'https://en.wiktionary.org' + a_tag.get('href')
                break
                
        current_url = next_link
        if current_url:
            time.sleep(0.1)  # small delay for pagination requests
            
    return links

def get_idiom_example(url, language):
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Determine the heading ID for the language
    if language == "Serbian(Serbo-Croatian)":
        lang_id = "Serbo-Croatian"
    else:
        lang_id = language
    
    # Find the language section - h2 now has the id directly (e.g. <h2 id="French">)
    lang_h2 = soup.find('h2', id=lang_id)
    
    # Fallback: try older format with child span
    if not lang_h2:
        lang_h2 = soup.find(lambda tag: tag.name == 'h2' and tag.find('span', id=lang_id))
    
    if not lang_h2:
        return None
        
    # Handle MediaWiki's newer mw-heading div wrappers
    if lang_h2.parent and lang_h2.parent.name == 'div' and 'mw-heading' in lang_h2.parent.get('class', []):
        traversal_start = lang_h2.parent
    else:
        traversal_start = lang_h2
        
    example_text = None
    
    # Check siblings under the language section until the next h2 (or equivalent)
    for sibling in traversal_start.find_next_siblings():
        # Stop if we hit the next language section heading
        if sibling.name == 'h2' or (sibling.name == 'div' and 'mw-heading2' in sibling.get('class', [])):
            break
            
        # Strategy 1: Look for e-example class (most common for pure text examples)
        examples = sibling.find_all(class_='e-example')
        if examples:
            example_text = examples[0].get_text(strip=True)
            break
            
        # Strategy 2: Look for h-usage-example div (often an italic phrase followed by translation)
        usage_divs = sibling.find_all('div', class_='h-usage-example')
        if usage_divs:
            # First try the 'i' tag, which customarily wraps the original text
            i_tag = usage_divs[0].find('i')
            if i_tag:
                example_text = i_tag.get_text(strip=True)
            else:
                # Fallback to the first line which contains the native sentence
                full_text = usage_divs[0].get_text(separator='\\n', strip=True)
                if full_text:
                    example_text = full_text.split('\\n')[0]
            break
            
        # Strategy 3: Look for quote blocks e-quotation
        quotes = sibling.find_all('div', class_='h-quotation')
        if quotes:
            quote_text_elem = quotes[0].find(class_='e-quotation')
            if quote_text_elem:
                example_text = quote_text_elem.get_text(strip=True)
            break
            
    return example_text

def main():
    results = {}
    
    for language, cat_url in URLS.items():
        print(f"\\nScraping category: {language}...")
        try:
            links = get_idiom_links(cat_url)
        except Exception as e:
            print(f"Failed to get links for {language}: {e}")
            continue
            
        random.shuffle(links)
        
        with_examples = []
        without_examples = []
            
        for i, (title, url) in enumerate(links):
            if len(with_examples) >= 50:
                print(f"  Reached 50 idioms with examples for {language}.")
                break
                
            print(f"  [{i+1}/{len(links)}] Checking idiom: {title} (Found examples: {len(with_examples)}/50)")
            try:
                example = get_idiom_example(url, language)
            except Exception as e:
                print(f"    Failed to get example for {title}: {e}")
                example = None
                
            if example:
                with_examples.append({
                    "idiom": title,
                    "example": example
                })
            else:
                without_examples.append({
                    "idiom": title,
                    "example": ""
                })
            
            time.sleep(0.1) # Small delay to not overwhelm Wiktionary
            
        lang_data = with_examples + without_examples
        lang_data = lang_data[:50]
        results[language] = lang_data
        
    output_path = 'idioms_output.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\\nScraping complete! Results saved to {output_path}")

if __name__ == "__main__":
    main()
