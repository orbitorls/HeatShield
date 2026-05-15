from datetime import date
import pandas as pd
from app.data.loaders import read_observations
from app.data.stations import STATIONS

def compute_hi(temp_c, rh):
    T = temp_c
    R = rh
    hi = (-8.78469475556 + 1.61139411 * T + 2.33854883889 * R +
          -0.14611605 * T * R + -0.012308094 * T**2 + -0.0164248277778 * R**2 +
          0.002211732 * T**2 * R + 0.00072546 * T * R**2 +
          -0.000003582 * T**2 * R**2)
    return hi

print('Station | Total Rows | Danger (>=42C) | Extreme (>=40C) | Months with Danger')
print('-' * 80)
for sid in STATIONS:
    df = read_observations(sid, date(2015, 1, 1), date(2024, 12, 31))
    if df.empty:
        print(f'{sid:8s} | NO DATA')
        continue
    df['ts_utc'] = pd.to_datetime(df['ts_utc'], utc=True)
    df['hi'] = compute_hi(df['temp_c'], df['rh'])
    danger = df[df['hi'] >= 42.0]
    extreme = df[df['hi'] >= 40.0]
    months = sorted(danger['ts_utc'].dt.month.unique()) if len(danger) > 0 else []
    print(f'{sid:8s} | {len(df):10d} | {len(danger):14d} | {len(extreme):15d} | {months}')
