import pandas as pd

dates = ['2021-06-15', '2022-06-15', '2023-04-15', '2023-12-15', '2024-06-15', '2024-12-15', '2025-03-15']
print('ERA5 Coverage Verification for BKK_01:')
print('='*60)

for date in dates:
    df = pd.read_parquet(f'data/raw/station_id=BKK_01/date={date}/obs.parquet')
    sources = df['source'].unique() if 'source' in df.columns else ['no_source']
    era5_count = df['source'].value_counts().get('era5', 0) if 'source' in df.columns else 0
    print(f'{date}: sources={sources}, ERA5 count={era5_count}')
