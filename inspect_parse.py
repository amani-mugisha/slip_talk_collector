"""inspect_parse.py — run from project root"""
import sys
sys.path.insert(0, '.')

from discovery.sitemap_fetcher.seeker.url_parser import parse
from discovery.sitemap_fetcher.models.urls_model import url_model

url = "https://example.com/path?q=1"
result = parse(url)

print("Type returned by parse():", type(result))
print("Attributes:", [a for a in dir(result) if not a.startswith('__')])
print()

# Show values of all attributes
for attr in [a for a in dir(result) if not a.startswith('__')]:
    try:
        val = getattr(result, attr)
        if not callable(val):
            print(f"  {attr} = {val!r}")
        else:
            print(f"  {attr}() = {val()!r}")
    except Exception as e:
        print(f"  {attr} → ERROR: {e}")
        