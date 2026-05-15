import json, os

base = r'D:\Heat-wave-backend\app\models\forecast_v3'
for sta, h in [('BKK_01', 6), ('BKK_01', 12), ('BKK_01', 24), ('CNX_01', 12), ('CNX_01', 24), ('HYI_01', 6), ('HYI_01', 12), ('HYI_01', 24), ('RYG_01', 6), ('RYG_01', 12), ('RYG_01', 24)]:
    reg_path = os.path.join(base, sta, f'h{h}', 'registry.json')
    if os.path.exists(reg_path):
        reg = json.load(open(reg_path))
        status = reg.get('status')
        mae = reg.get('mae')
        skill = reg.get('skill_score')
        pi = reg.get('prediction_interval', {})
        width = pi.get('mean_width') if pi else None
        cov = pi.get('coverage_90') if pi else None
        print(f'{sta}/h{h}: status={status} mae={mae} skill={skill} pi_width={width} pi_cov={cov}')
