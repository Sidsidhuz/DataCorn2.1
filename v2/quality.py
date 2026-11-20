"""Transparent heuristics. Suspects require domain review, never proof of leakage."""
import re
import numpy as np
import pandas as pd
from .data import fingerprint


def profile(df, target=None):
    columns, risks = [], []
    for c in df:
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s)
        columns.append({'column':c,'type':str(s.dtype),'missing':int(s.isna().sum()),
                        'missing_pct':round(float(s.isna().mean()*100),2),'unique':int(s.nunique())})
        if c == target:
            continue
        reason = []
        if s.nunique() <= 1:
            reason.append('Constant or empty feature')
        if re.search(r'(^|_)(id|uuid|key|index)($|_)',c.lower()):
            reason.append('Identifier: exclude from prediction')
        if not numeric and s.nunique() > min(100, len(df)*.8):
            reason.append('High-cardinality text / possible identifier')
        if re.search(r'closed|cancelled|canceled|post_outcome|after_|refund|outcome',c.lower()):
            reason.append('May only be available after the outcome')
        if target and df[target].nunique()>1:
            pairs = df[[c,target]].dropna()
            if len(pairs)>10 and pairs[c].nunique()>1:
                if (pairs[c].astype(str).to_numpy()==pairs[target].astype(str).to_numpy()).all():
                    reason.append('Exact target copy')
                elif numeric and pd.api.types.is_numeric_dtype(df[target]):
                    corr = pairs[c].corr(pairs[target])
                    if np.isfinite(corr) and abs(corr)>.995:
                        reason.append('Near-perfect target correlation; review timing')
                elif pairs[c].nunique()<=20 and pairs[c].nunique()<=pairs[target].nunique():
                    if pairs.groupby(c)[target].nunique().max()==1:
                        reason.append('Deterministic target mapping')
        if reason:
            risks.append({'column':c,'reasons':reason,
                          'exclude':any(not r.startswith('Near-perfect') for r in reason)})
    return {'rows':len(df),'columns':columns,'duplicates':int(df.duplicated().sum()),
            'fingerprint':fingerprint(df),'risks':risks,
            'caveat':'Heuristics cannot establish feature availability or detect every leak. Review collection timing and repeated entities.'}
