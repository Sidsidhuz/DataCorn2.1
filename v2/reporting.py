"""Reports are assembled from computed evidence; no invented numerical prose."""
from io import BytesIO
from xml.sax.saxutils import escape
import json
from importlib.metadata import version
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill


def evidence_package(finding, challenge, quality, model=None):
    result={'finding':finding,'challenge':challenge,'quality':quality,
            'dependencies':{name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','shap','streamlit']}}
    if model:
        result['model']={k:model[k] for k in ['winner','task','target','features','excluded_features',
            'leaderboard','holdout','cv_metric','train_rows','test_rows','missing_target_rows',
            'duplicates_removed','seed','limitations']}
        result['model']['parameters']={k:str(v) for k,v in model['pipeline'].get_params().items()}
        if 'split' in model:
            result['model']['split']=model['split']
    return result


def json_report(package):
    return json.dumps(package,indent=2,ensure_ascii=False,allow_nan=False,default=str).encode('utf-8')


def pdf_report(package):
    output=BytesIO(); styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Eyebrow',fontSize=9,textColor=colors.HexColor('#0D776E'),spaceAfter=12))
    styles['Title'].textColor=colors.HexColor('#172E35')
    styles['BodyText'].leading=15
    story=[]
    def para(value,style='BodyText'):
        return Paragraph(escape(str(value)),styles[style])
    def heading(title):
        story.extend([Spacer(1,14),para(title,'Heading2')])
    f,c,q=package['finding'],package['challenge'],package['quality']
    story.extend([para('DATACORN 2.1 / EVIDENCE BRIEF','Eyebrow'),
                  para('The finding. And the challenge.','Title'),
                  para(f['plan']['question']),Spacer(1,12),para('Verdict: '+c['verdict'],'Heading2'),
                  para(c['recommended_action'])])
    heading('01 / What the data says')
    for line in [f['finding'],'Test: '+f['test'],f"Complete observations: {f['n']}; excluded rows: {f['excluded_rows']}",
                 '95% interval: '+(f"[{f['ci95'][0]:.4f}, {f['ci95'][1]:.4f}]" if f['ci95'] else 'Not supplied for this effect'),f['scope']]:
        story.append(para(line))
    if f.get('groups'):
        rows=[['Group','n','Mean']]+[[g['group'],str(g['n']),f"{g['mean']:.4f}"] for g in f['groups']]
        table=Table([[para(cell) for cell in row] for row in rows],colWidths=[240,70,140],repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7F1ED')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8)]))
        story.extend([Spacer(1,10),table])
    heading('02 / Challenge My Finding')
    for check in c['checks']:
        story.append(KeepTogether([para(check['check']+' / '+check['status'].upper(),'Heading3'),para(check['detail'])]))
    if c['strata']:
        heading('03 / Within-segment evidence')
        for row in c['strata']:
            story.append(para(f"{row['stratum']}: n={row['n']}, effect={row['effect']:.4f}, p={row['p_value']:.4g}, Holm-adjusted p={row['holm_p']:.4g}"))
    if package.get('model'):
        m=package['model'];heading('04 / Predictive model')
        for line in [f"CV winner: {m['winner']}; selection metric: {m['cv_metric']}",
                     f"Train rows: {m['train_rows']}; holdout rows: {m['test_rows']}",
                     'Final holdout metrics: '+', '.join(f'{k}={v:.3f}' for k,v in m['holdout'].items()),
                     'Retained predictors: '+', '.join(m['features'])]:
            story.append(para(line))
        for line in m['limitations']:
            story.append(para('- '+line))
    heading('Limitations and next decision')
    for line in c['limitations']+c['skipped_strata']:
        story.append(para('- '+line))
    heading('Reproduce this investigation')
    for line in ['Dataset SHA-256: '+q['fingerprint'],f"Engine: {f['engine_version']}; random seed: {f['seed']}",
                 'Engine SHA-256: '+f['engine_sha256'],
                 'Created (UTC): '+f['created_at'],'Planner: '+f['plan'].get('source','User confirmed'),
                 'Download the companion JSON for the complete plan, checks, model settings and evidence.']:
        story.append(para(line))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#617377'))
        canvas.drawString(44,26,'DataCorn 2.1 | Exploratory evidence, not causal proof')
        canvas.drawRightString(A4[0]-44,26,str(doc.page))
    SimpleDocTemplate(output,pagesize=A4,rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    return output.getvalue()


def excel_report(package):
    wb=Workbook();ws=wb.active;ws.title='Evidence'
    f=package['finding']
    for row in [('Field','Computed value'),('Question',f['plan']['question']),('Finding',f['finding']),
                ('Test',f['test']),('p-value',f['p_value']),('Effect',f['effect']),
                ('Verdict',package['challenge']['verdict']),('Dataset SHA256',f['dataset_sha256'])]:
        ws.append(row)
    checks=wb.create_sheet('Challenge');checks.append(['Check','Status','Evidence'])
    for r in package['challenge']['checks']:
        checks.append([r['check'],r['status'],r['detail']])
    strata=wb.create_sheet('Strata');strata.append(['Stratum','n','Effect','p-value','Holm p'])
    for r in package['challenge']['strata']:
        strata.append([r[k] for k in ['stratum','n','effect','p_value','holm_p']])
    for sheet in wb:
        sheet.freeze_panes='A2';sheet.column_dimensions['A'].width=32;sheet.column_dimensions['B'].width=54;sheet.column_dimensions['C'].width=70
        for row in sheet:
            for cell in row:
                # Uploaded text cannot become an Excel formula.
                if isinstance(cell.value,str) and cell.value.lstrip().startswith(('=','+','-','@')):
                    cell.value="'"+cell.value
        for cell in sheet[1]:
            cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='0D776E')
    out=BytesIO();wb.save(out);return out.getvalue()
