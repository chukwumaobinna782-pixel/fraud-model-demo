# simulator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

CATS = {
    'merchant_category': ['grocery', 'gas', 'restaurant', 'online_retail', 'atm'],
    'device_os':         ['iOS', 'Android', 'Windows', 'macOS', 'Linux', 'ChromeOS'],
    'device_lang':       ['en-US', 'es-US', 'fr-US', 'zh-CN'],
    'ip_country':        ['US', 'CA', 'RU', 'CN', 'KP', 'IR', 'VE', 'BR', 'IN'],
    'ip_isp':            ['comcast', 'verizon', 'aws', 'azure', 'china_telecom']
}

def gen_tx(customer_id=None, fraud=False):
    """Return a single-row DataFrame that mimics one transaction."""
    rng = np.random.default_rng()

    now = datetime.utcnow()
    hour = rng.integers(0, 24)
    minute = rng.integers(0, 60)
    ts = now.replace(hour=hour, minute=minute, second=rng.integers(0, 60))

    # ----- amount -----
    if fraud:
        amount = rng.lognormal(mean=4.2, sigma=1.4)   # higher, more volatile
    else:
        amount = rng.lognormal(mean=3.1, sigma=0.8)
    amount_usd = np.round(amount, 2)

    # ----- location -----
    dist_to_home = rng.exponential(12) if not fraud else rng.exponential(50)
    dist_to_home = np.round(dist_to_home, 1)

    # ----- categorical -----
    merch_cat = rng.choice(CATS['merchant_category'])
    os = rng.choice(CATS['device_os'])
    lang = rng.choice(CATS['device_lang'])

    if fraud:
        ip_country = rng.choice(CATS['ip_country'], p=[.10, .05, .15, .15, .10, .10, .10, .15, .10])
    else:
        ip_country = rng.choice(CATS['ip_country'], p=[.65, .10, .02, .03, .01, .01, .01, .12, .05])
    isp = rng.choice(CATS['ip_isp'])

    # ----- booleans -----
    card_present = int(rng.random() > (.7 if fraud else .2))
    foreign_ip = int(ip_country != 'US')
    risky_ip_country = int(ip_country in ['RU', 'CN', 'KP', 'IR', 'VE'])
    uncommon_os = int(os in ['Linux', 'ChromeOS'])

    # ----- velocity -----
    cust_prev_tx = rng.integers(0, 300)
    time_since_prev = rng.exponential(900) if not fraud else rng.exponential(90)
    very_quick = int(time_since_prev < 300)
    hist_avg = amount_usd * rng.lognormal(0, 0.2)
    ratio = amount_usd / (hist_avg + 1)

    row = {
        'trans_ts': ts,
        'customer_id': customer_id or rng.integers(1_000_000, 9_999_999),
        'amount_usd': amount_usd,
        'dist_to_home_km': dist_to_home,
        'merchant_category': merch_cat,
        'device_os': os,
        'device_lang': lang,
        'ip_country': ip_country,
        'ip_isp': isp,
        'card_present': card_present,
        'foreign_ip': foreign_ip,
        'risky_ip_country': risky_ip_country,
        'uncommon_os': uncommon_os,
        'cust_prev_tx': cust_prev_tx,
        'time_since_prev_tx_sec': time_since_prev,
        'very_quick_succession': very_quick,
        'cust_hist_avg_amount': hist_avg,
        'amount_to_hist_avg_ratio': ratio,
        'is_fraud': int(fraud)
    }
    return pd.Series(row)
