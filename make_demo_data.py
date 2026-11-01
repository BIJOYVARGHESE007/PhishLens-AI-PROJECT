"""Creates a SMALL SYNTHETIC dataset so you can test the pipeline end-to-end.
For your real project/placement, use a real dataset (see README) and put it at data/urls.csv."""
import random, csv
random.seed(7)
brands = ["paypal", "sbi", "hdfc", "amazon", "google", "netflix", "icici", "microsoft", "apple", "flipkart"]
good_hosts = ["wikipedia.org", "github.com", "python.org", "bbc.com", "nytimes.com", "stackoverflow.com",
              "coursera.org", "mit.edu", "nptel.ac.in", "gov.in", "amazon.in", "flipkart.com", "hdfcbank.com"]
good_paths = ["", "/", "/about", "/docs/intro", "/wiki/Machine_learning", "/courses/python", "/news/world", "/help/contact"]
bad_tld = ["xyz", "top", "tk", "click", "support", "work"]
kw = ["login", "verify", "secure", "update", "account", "confirm", "signin"]
rows = []
for _ in range(1500):
    h, pth = random.choice(good_hosts), random.choice(good_paths)
    extra = random.choice(["", "", f"/{random.choice(['article','post','item','page'])}-{random.randint(1,99999)}",
                           f"/account/login?ref={random.randint(1,999)}", "/secure/checkout", f"/user/{random.randint(1,9999)}/settings"])
    rows.append((f"{random.choice(['https','https','http'])}://{random.choice(['www.','','accounts.','mail.'])}{h}{pth}{extra}", 0))
for _ in range(1500):
    b, k, t = random.choice(brands), random.choice(kw), random.choice(bad_tld)
    style = random.randint(0, 4)
    if style == 0: u = f"http://{b}-{k}-{random.randint(100,999)}.{t}/{k}/index.php?id={random.randint(1000,99999)}"
    elif style == 1: u = f"http://{random.randint(11,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}/{b}/{k}.html"
    elif style == 2: u = f"https://{b}.{k}.secure-{random.randint(1,99)}.{t}/account/verify?user={random.randint(10000,99999)}&token=%2F%3D%3D"
    elif style == 3: u = f"http://{b}.com@{random.choice(['evil','login','free'])}-{random.randint(1,999)}.{t}/{k}"
    else: u = f"https://{b}{random.choice(['-help','-care','online','-portal'])}{random.randint(1,99)}.com/{k}"
    rows.append((u, 1))
random.shuffle(rows)
with open("data/urls.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["url", "label"]); w.writerows(rows)
print("Wrote data/urls.csv (SYNTHETIC) with", len(rows), "rows")
