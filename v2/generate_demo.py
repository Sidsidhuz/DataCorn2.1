"""Generate inspectable artifacts from the same functions used by the UI."""
from pathlib import Path
import json
from v2.data import demo_data
from v2.planner import propose
from v2.quality import profile
from v2.statistics import analyze
from v2.challenge import challenge
from v2.modeling import train
from v2.reporting import evidence_package,json_report,pdf_report,excel_report

if __name__=='__main__':
    output=Path('v2/demo_output');output.mkdir(exist_ok=True)
    df=demo_data();plan=propose('Does premium support increase customer churn?',df)
    finding=analyze(df,plan);quality=profile(df,'churn')
    review=challenge(df,finding,quality,'customer_segment')
    model=train(df,'churn','classification')
    package=evidence_package(finding,review,quality,model)
    (output/'evidence.json').write_bytes(json_report(package))
    (output/'evidence.pdf').write_bytes(pdf_report(package))
    (output/'evidence.xlsx').write_bytes(excel_report(package))
    df.to_csv(output/'demo_customers.csv',index=False)
    print(json.dumps({'finding':finding['finding'],'verdict':review['verdict'],
          'strata':review['strata'],'winner':model['winner'],'holdout':model['holdout']},indent=2))
