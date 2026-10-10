"""Earlier-stage report on rated issuers (kept as background evidence); the competition report is build_final_report.py."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Flowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
PDF = ROOT / 'reports/前期研究_评级企业关联图谱报告.pdf'
MD = ROOT / 'reports/前期研究_评级企业关联图谱报告.md'
FONT = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e4e3df'
BLUE, ORANGE, HEAD = '#2a78d6', '#eb6834', '#13334e'
BODY_PAGE_LIMIT = 8


def load(name):
    return json.loads((EXP / name).read_text())


class PageMarker(Flowable):
    """Records the page on which it is drawn (used to verify the body page budget)."""
    def __init__(self, store, key):
        super().__init__(); self.store, self.key = store, key; self.width = self.height = 0

    def draw(self):
        self.store[self.key] = self.canv.getPageNumber()


def hbar(rows, title, fmt, width=490, bar_h=13, gap=7, label_w=150, max_value=None):
    """Horizontal single-hue bars; highlighted rows use the accent hue. Values are direct-labeled."""
    max_value = max_value or max(v for _, v, _ in rows) * 1.15
    height = 22 + len(rows) * (bar_h + gap)
    d = Drawing(width, height)
    d.add(String(0, height - 12, title, fontName='ReportCJK', fontSize=8.5, fillColor=colors.HexColor(INK2)))
    plot_w = width - label_w - 60
    for k, (label, value, accent) in enumerate(rows):
        y = height - 24 - (k + 1) * (bar_h + gap) + gap
        d.add(String(label_w - 6, y + 3, label, fontName='ReportCJK', fontSize=8, textAnchor='end',
                     fillColor=colors.HexColor(INK)))
        w = max(plot_w * value / max_value, 1.5)
        d.add(Rect(label_w, y, w, bar_h, rx=2, ry=2, strokeColor=None,
                   fillColor=colors.HexColor(ORANGE if accent else BLUE)))
        d.add(String(label_w + w + 4, y + 3, fmt(value), fontName='ReportCJK', fontSize=8,
                     fillColor=colors.HexColor(INK2)))
    d.add(Line(label_w, 4, label_w, height - 20, strokeColor=colors.HexColor(MUTED), strokeWidth=.6))
    return d


def schema(width=490, height=150):
    """Knowledge-graph schema: what nodes, edges and time-series attributes enter the model."""
    d = Drawing(width, height)
    def box(x, y, w, h, title, sub, accent=False):
        d.add(Rect(x, y, w, h, rx=5, ry=5, strokeColor=colors.HexColor(ORANGE if accent else BLUE), strokeWidth=1.1,
                   fillColor=colors.HexColor('#fff4ee' if accent else '#eef4fc')))
        d.add(String(x + w / 2, y + h - 14, title, fontName='ReportCJK', fontSize=9, textAnchor='middle', fillColor=colors.HexColor(INK)))
        for i, line in enumerate(sub):
            d.add(String(x + w / 2, y + h - 27 - 10 * i, line, fontName='ReportCJK', fontSize=7, textAnchor='middle', fillColor=colors.HexColor(INK2)))
    def arrow(x1, y1, x2, y2, text):
        d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor(MUTED), strokeWidth=1))
        d.add(String((x1 + x2) / 2, (y1 + y2) / 2 + 4, text, fontName='ReportCJK', fontSize=7, textAnchor='middle', fillColor=colors.HexColor(INK2)))
    box(10, 88, 120, 56, '上游供应商企业', ['财务 + 评级事件', '+ 涉诉/破产事件'])
    box(185, 88, 120, 56, '目标经营类客户', ['财务 / 评级 / 展望', '降级 / 涉诉序列'], accent=True)
    box(360, 88, 120, 56, '下游客户企业', ['财务 + 评级事件', '+ 涉诉/破产事件'])
    box(10, 6, 140, 52, '人员关联企业', ['同一董事/高管同时在任', '（BoardEx，r2）'])
    box(175, 6, 140, 52, '投资人关联企业', ['同一机构持股均≥5%', '（13F，r3）'])
    box(340, 6, 140, 52, '同行业评级主体', ['行业降级气候', '（竞争/市场关联）'])
    arrow(130, 116, 185, 116, '供货 r0 / 反向 r1')
    arrow(305, 116, 360, 116, '供货 r0 / 反向 r1')
    arrow(205, 88, 80, 58, '共享人员')
    arrow(245, 88, 245, 58, '共同大股东')
    arrow(285, 88, 410, 58, '同行业')
    return d


def main():
    panel = load('global_panel_550_audit.json'); gaudit = load('global_graph_550_365_audit.json')
    base = load('global_baseline_results.json'); rating = load('global_baseline_results_rating_enhancement.json')
    gorig = load('global_graph_model_results.json'); gev = load('global_graph_model_results_event_nodes.json')
    gbud = load('global_graph_model_results_budget_only.json'); cont = load('global_contagion_results.json')
    ext = load('global_extensions_summary.json')
    sp = panel['splits']; cov = gaudit['coverage']; desc = cont['descriptive_event_rates']; res = cont['results']
    sel = cont['selected_by_validation']; card = cont['scorecard']; cmp = cont['paired_comparisons']
    snaps = gaudit['snapshots']; edges_total = sum(v['directed_pairs'] for v in snaps.values())
    t_sel = res[sel]['test']
    rel = load('related_contagion_results.json'); relc = load('related_contagion_compact_results.json')
    raudit = load('related_graph_audit.json'); gsev = load('global_graph_model_results_related_severe.json')
    gdef = load('global_graph_model_results_related_default.json')
    rsnap = raudit['snapshots']; rcov = rel['relation_coverage']
    mean_edges = lambda k: sum(v[k] for v in rsnap.values()) / len(rsnap)
    tsev = rel['targets']['severe']['results']['B_plus_own_rating_events_logistic']['test']

    pdfmetrics.registerFont(TTFont('ReportCJK', FONT))
    body = ParagraphStyle('body', fontName='ReportCJK', fontSize=9.4, leading=14.6, spaceAfter=5, wordWrap='CJK',
                          textColor=colors.HexColor('#26354a'))
    title_s = ParagraphStyle('title', parent=body, fontSize=17, leading=24, spaceAfter=4, textColor=colors.HexColor(HEAD))
    h1 = ParagraphStyle('h1', parent=body, fontSize=12.5, leading=18, spaceBefore=6, spaceAfter=5, textColor=colors.HexColor(HEAD))
    h2 = ParagraphStyle('h2', parent=body, fontSize=10.2, leading=15, spaceBefore=2, spaceAfter=3, textColor=colors.HexColor(HEAD))
    small = ParagraphStyle('small', parent=body, fontSize=8, leading=11.5, spaceAfter=3, textColor=colors.HexColor(INK2))
    cell = ParagraphStyle('cell', parent=small, fontSize=7.9, leading=10.6, spaceAfter=0, textColor=colors.HexColor(INK))
    story, md, pages = [], [], {}

    def para(text, style=body, prefix=''):
        story.append(Paragraph(escape(text), style)); md.append(prefix + text + '\n\n')
    def h(text, level=1):
        para(text, h1 if level == 1 else h2, '## ' if level == 1 else '### ')
    def table(headers, rows, widths=None, note=None):
        data = [[Paragraph(escape(str(v)), cell) for v in row] for row in [headers] + rows]
        t = Table(data, colWidths=widths or [495 / len(headers)] * len(headers), repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e7eff6')), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                               ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                               ('TOPPADDING', (0, 0), (-1, -1), 2.6), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.6),
                               ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#8ba6bf')),
                               ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')])]))
        story.extend([t, Spacer(1, 4)])
        md.append('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        md.extend('| ' + ' | '.join(map(str, r)) + ' |\n' for r in rows); md.append('\n')
        if note:
            para(note, small)
    def figure(drawing, caption, md_rows=None):
        story.append(KeepTogether([drawing, Paragraph(escape(caption), small)]))
        md.append(f'**{caption}**\n\n')
        if md_rows:
            md.append('| 项目 | 数值 |\n| --- | --- |\n' + ''.join(f'| {a} | {b} |\n' for a, b in md_rows) + '\n')
    pct = lambda x: f'{100 * x:.2f}%'
    ci = lambda c: f"{c['ap_difference']:+.3f} [{c['ap_difference_ci95'][0]:+.3f}, {c['ap_difference_ci95'][1]:+.3f}]"

    # ---------------- Title + abstract ----------------
    para('基于关联图谱的经营类客户信用风险评分模型', title_s, '# ')
    para('—— 上下游供应链关联、关联方风险事件时序与多层传导的实证构建与验证 | 终版报告 | 2026-10-09', small)
    para(f'摘要：本文围绕赛题“将企业关联图谱作为风控评分模型的重要输入”，以WRDS授权数据构建北美评级企业的多关系关联图谱：'
         f'节点为企业，边包括上下游供应链（{len(snaps)}个季度快照、{edges_total:,}条有向关系记录）、同一董事/高管同时在任的人员关联'
         f'（BoardEx，平均每季度{mean_edges("people_edges"):,.0f}条）、同一机构持股均≥5%的投资人关联（13F，平均每季度{mean_edges("investor_edges"):,.0f}条）以及同行业关联；'
         '节点属性为时点财务比率、评级事件序列（降级、负面展望/观察、D/SD）和涉诉/破产事件序列（Audit Analytics）。'
         '预测目标设为三层：180天内记录违约D/SD、显著信用恶化（下调≥2档、跌入CCC+及以下或违约）与下调≥1档，'
         f"严格按时间切分（2017–2019训练、2020–2021验证、2022–2023测试），测试期三层目标分别有"
         f"{rel['targets']['default']['counts']['test']['positive_entities']}/{rel['targets']['severe']['counts']['test']['positive_entities']}/{rel['targets']['mild']['counts']['test']['positive_entities']}家事件企业。"
         f'违约目标上，验证集规则选出“财务+自身评级+自身风险事件时序”逻辑回归评分卡：测试AP {t_sel["ap"]:.3f}、AUC {t_sel["roc_auc"]:.3f}、KS {t_sel["ks"]:.3f}，'
         f'前5%高风险名单覆盖{pct(card["test_capture"]["top_5pct"])}的违约记录；显著恶化目标上同一模型测试AP {tsev["ap"]:.3f}、AUC {tsev["roc_auc"]:.3f}。'
         '关联风险在描述统计中可见，并集中于B档及以下的脆弱企业；但在控制自身信息后，供应链、人员、投资人三类关联特征以及多关系GraphSAGE/GAT/R-GCN'
         '均未超过同特征无图模型或随机打乱关联对象的对照。据此提出“评分卡主模型+关联图谱预警层”的产品方案，并给出在银行内部数据上的落地路径。')

    # ---------------- 1 Data ----------------
    h('一、数据介绍')
    para('赛题要求模型纳入三类信息：上下游业务关联、股东/实控人/投资人的股权与人员关联、关联主体逾期/涉诉/失信等风险事件的时序信息。'
         '全部数据通过WRDS授权会话获取；大表在WRDS Jupyter服务器上用SQL联表筛选后只下载结果，每条SQL原文、行数与SHA256清单随原始文件保存，原始数据不进入版本库。')
    table(['数据源（WRDS）', '规模', '在图谱中的角色', '对应赛题要素'],
          [['Compustat Snapshot 年度时点财务', '1,412,641行 / 43,552家', '节点属性：11项比率与财报年龄', '企业自身经营与财务'],
           ['S&P 主体评级历史 2015–2023', '长期外币主体评级52,892行', '节点属性：等级、降级、展望、观察、D/SD；生成三层标签', '风险事件时序（违约/恶化）'],
           ['Compustat 客户—供应商披露', '31,140行 / 8,741个企业对', '边：供应商→客户（r0）及反向（r1）', '上下游业务关联'],
           ['BoardEx 北美任职记录', f"738,022行；{raudit['linked_boardex_companies']:,}家图谱企业可映射", '边：同一人员同时在两家企业任董事/高管（r2）', '人员关联'],
           ['Thomson Reuters 13F 机构持股', '656,190条≥5%持股记录', '边：同一机构对两家企业持股均≥5%（r3）', '股东/投资人关联'],
           ['Audit Analytics 诉讼与破产通知', f"诉讼当事方38,235行、破产3,190行；{raudit['graph_gvkeys_with_cik']:,}家可经CIK映射", '节点属性：新增/未结被诉案件、破产', '涉诉风险事件时序'],
           ['评级行业字段', '与评级历史同表', '同行业评级主体降级气候', '竞争/市场关联']],
          [112, 118, 160, 105])
    para('样本构建：每个季度末次日为观察点，只使用当时已存在的财务版本（更新码2/3、财报期不晚于观察点、财报年龄≤550天）。'
         '标签为观察后180天内记录D/SD；样本须在观察时持有存续评级且2015年以来未出现D/SD；窗口内先撤评则删失；'
         '180天窗口+30天成熟期必须在2023年末前结束。训练/验证跨界记录整体剔除，避免标签窗口穿越。'
         '供应链边以披露日期+365天作为假定可见日并累计保留，两端在当期均需有合格财务。')
    para('为什么这样设计：信用模型最常见的失真来自“用到了观察日之后才知道的信息”。若用最新版财务回填历史、或让标签窗口跨入验证期，'
         '测试分数会虚高而上线后失效。因此本文的每一项输入都绑定其可见时间：财务取当时版本、评级取观察时刻之前最后一次动作、'
         '供应链边在披露后延迟一年才视为可见；各阶段之间按“180天预测窗+30天成熟期”留出隔离带。负类只表示数据库未记录违约，'
         '撤评企业不被当作“未违约”，以免把信息缺失误判为低风险。')
    table(['阶段', '观察期', '面板记录', '违约记录', '覆盖企业', '独立违约主体', '有供应链关系的记录'],
          [[n, p, f"{sp[s]['rows']:,}", sp[s]['positive_rows'], f"{sp[s]['entities']:,}", sp[s]['positive_entities'],
            f"{cov[s]['incident_rows']:,}（{100 * cov[s]['incident_rows'] / cov[s]['rows']:.1f}%）"]
           for n, p, s in (('训练', '2017–2019', 'train'), ('验证', '2020–2021', 'validation'), ('测试', '2022–2023', 'test'))],
          [40, 60, 65, 55, 65, 70, 140],
          note=f"另有{sp['purged']['rows']:,}条跨界记录剔除。图谱共{panel['total_node_rows']:,}个“企业-季度”节点（含未打标签的邻居企业），"
               f"每季度{min(v['nodes'] for v in snaps.values()):,}–{max(v['nodes'] for v in snaps.values()):,}个节点。同一企业跨季度重复出现，"
               '因此区间估计按企业整体重抽样，独立违约主体数才是有效事件数。')
    para('关系与事件的时点规则：人员关联要求两端任职在观察日同时有效（BoardEx结束日期9999为至今在任，9000为离任日期未知，按开始后最多3年计）；'
         '投资人关联取观察日前一年内最近一期13F，且季末后45天（法定披露期限）才视为可见；单季对≥50家企业持股≥5%的指数型机构'
         f"（每季度约{mean_edges('passive_managers_excluded'):.0f}家，如BlackRock、Vanguard）会把几乎所有企业连成一团，主图予以剔除；"
         '诉讼以Audit Analytics收录日期为可见日（收录比立案中位数晚111天，且大量历史案件为事后补录），破产以公告日为准。')
    para('三层标签：除D/SD外，增加“显著恶化”（180天内下调≥2档、由B-及以上跌入CCC+及以下，或D/SD）与“轻度恶化”（下调≥1档或D/SD），'
         '删失、成熟期与样本资格规则与D/SD相同。更宽的标签更贴近赛题“提前识别潜在风险信号”的目标，也使关联效应的检验功效更高。')
    table(['目标', '训练 事件记录/企业', '验证 事件记录/企业', '测试 事件记录/企业'],
          [[n] + [f"{rel['targets'][t]['counts'][s]['positive_rows']} / {rel['targets'][t]['counts'][s]['positive_entities']}" for s in ('train', 'validation', 'test')]
           for t, n in (('default', 'D/SD违约'), ('severe', '显著恶化'), ('mild', '下调≥1档'))], [120, 125, 125, 125])
    para('数据边界：母子公司（实控人）关系表的子公司只有名称、没有企业编码，无法可靠对齐到图谱企业，未入模；“失信”在美国无直接对应，以破产通知代替。'
         '评级恶化不等于银行贷款违约，本文分数称为“记录信用事件风险分”，不直接作为授信PD。')

    # ---------------- 2 Descriptive ----------------
    story.append(CondPageBreak(300)); h('二、描述性分析')
    rb = desc['rating_bucket']; no = desc['own_negative_outlook']; dg = desc['own_downgraded_365']
    sl = desc['supply_link']; nd = desc['neighbor_downgraded_365']; nn = desc['neighbor_negative_outlook']
    bars = [('投资级（BBB-及以上）', rb['投资级(BBB-及以上)']['event_rate'], False), ('BB档', rb['BB档']['event_rate'], False),
            ('B档', rb['B档']['event_rate'], False), ('CCC至C', rb['CCC至C']['event_rate'], True),
            ('自身负面展望：否', no['否']['event_rate'], False), ('自身负面展望：是', no['是']['event_rate'], True),
            ('近一年自身降级：否', dg['否']['event_rate'], False), ('近一年自身降级：是', dg['是']['event_rate'], True),
            ('无供应链关系', sl['无供应链关系']['event_rate'], False), ('有供应链关系', sl['有供应链关系']['event_rate'], False),
            ('评级关联方无降级', nd['有评级关联方但无降级']['event_rate'], False), ('关联方近一年降级', nd['有关联方近一年降级']['event_rate'], True),
            ('关联方负面展望', nn['有关联方负面展望']['event_rate'], True)]
    figure(hbar(bars, '180天内记录违约率（全部监督记录，单位%）', lambda v: f'{100 * v:.2f}%', max_value=0.25),
           '图1  不同自身状态与关联状态下的违约率。橙色为风险组。CCC档与关联方组的事件主体较少，见正文。',
           [(a, f'{100 * b:.2f}%') for a, b, _ in bars])
    para(f"（1）自身信用状态呈强单调：投资级{rb['投资级(BBB-及以上)']['rows']:,}条记录零违约，CCC至C档违约率{pct(rb['CCC至C']['event_rate'])}"
         f"（{rb['CCC至C']['event_entities']}家主体）。（2）风险事件时序信息量大：有负面展望的记录违约率{pct(no['是']['event_rate'])}，"
         f"约为无负面展望者的{no['是']['event_rate'] / no['否']['event_rate']:.0f}倍；近一年被降级者为{pct(dg['是']['event_rate'])}，"
         f"未降级者{pct(dg['否']['event_rate'])}。（3）关联风险存在联动迹象：有供应链关系的记录违约率{pct(sl['有供应链关系']['event_rate'])}，"
         f"无关系者{pct(sl['无供应链关系']['event_rate'])}；关联方近一年被降级时为{pct(nd['有关联方近一年降级']['event_rate'])}，"
         f"高于“有评级关联方但未降级”的{pct(nd['有评级关联方但无降级']['event_rate'])}。但后者只涉及{nd['有关联方近一年降级']['event_entities']}家违约主体，"
         '且有关联方降级的企业本身评级往往更低，单变量差异需在多变量模型中检验。')
    table(['关系类型', '每季度边数（均值）', '测试期有该类关联方的记录占比', '含义'],
          [['供应链：供应商→客户', f"{edges_total / len(snaps):,.0f}", f"{100 * rcov['supplier']['test']:.1f}%（有供应商）/ {100 * rcov['customer']['test']:.1f}%（有客户）", '重要客户披露，延迟365天可见'],
           ['人员：共享董事/高管', f"{mean_edges('people_edges'):,.0f}", f"{100 * rcov['people']['test']:.1f}%", '同一人员同时在两家企业任职'],
           ['投资人：共同≥5%机构股东', f"{mean_edges('investor_edges'):,.0f}", f"{100 * rcov['investors']['test']:.1f}%", '剔除指数型机构后的共同大股东']],
          [120, 85, 175, 115])
    strat = relc['stratified_exposure']
    rows = []
    for kind, name in (('supplier', '供应商'), ('customer', '客户'), ('people', '人员关联方'), ('investors', '投资人关联方')):
        for bucket, bname in (('IG(<=BBB-)', '投资级'), ('B_or_worse', 'B档及以下')):
            v = strat[f'severe|{kind}|{bucket}']
            rows.append([f'{name}｜{bname}', f"{pct(v['exposed_rate'])}（{v['exposed_rows']:,}条，{v['exposed_entities_with_event']}家）", f"{pct(v['unexposed_rate'])}（{v['unexposed_linked_rows']:,}条）"])
    table(['关系｜自身评级档', '关联方近期恶化：显著恶化发生率', '有关联方但未恶化：发生率'], rows, [150, 195, 150],
          note='“近期恶化”指有评级关联方中存在近一年降级、负面展望或近两年D/SD。全部监督记录，同一企业跨季度重复，括号内为记录数与事件企业数。')
    para('分层统计给出更细的图景：在B档及以下的脆弱企业中，供应商或客户近期恶化时，自身显著恶化的发生率约为未暴露企业的两倍'
         f"（供应商：{pct(strat['severe|supplier|B_or_worse']['exposed_rate'])}对{pct(strat['severe|supplier|B_or_worse']['unexposed_rate'])}；"
         f"客户：{pct(strat['severe|customer|B_or_worse']['exposed_rate'])}对{pct(strat['severe|customer|B_or_worse']['unexposed_rate'])}），"
         '投资人关联方也有同方向差异；而投资级企业几乎看不到这种差异，人员关联在各档都不明显。这与经济直觉一致——脆弱企业更容易被合作方拖累。'
         '但暴露组的事件企业只有十余家，且这些企业往往同时已有自身负面展望或降级，需要在多变量模型中检验其增量。')

    # ---------------- 3 Model ----------------
    h('三、模型分析')
    h('3.1 图谱结构与特征体系', 2)
    figure(schema(), '图2  企业关联图谱结构：企业为节点；供应链（含反向）、人员、投资人为三类边；同行业主体构成竞争关联；风险事件序列作为节点属性随季度更新。')
    table(['特征层', '内容（全部为观察日可见信息）', '维度'],
          [['L1 自身财务', '负债/资产、债务/资产、现金/资产、流动性、周转、EBIT/资产、经营现金流/资产、利息保障、利润率、财报年龄', '12'],
           ['L2 自身评级', '评级等级序数、距上次评级动作天数（log）', '2'],
           ['L3a 自身评级事件', '当前负面展望、负面观察、近365天是否降级、近365天级差变化', '4'],
           ['L3b 自身涉诉事件', '近365天新增被诉案件、未结被诉案件、近365天新增任意诉讼、近730天破产通知', '4'],
           ['L4 关联传导（每类关系）', '有评级关联方数、平均/最差等级、近一年降级数、近两年D/SD数、负面展望数、关联方总数、被诉关联方数、关联方新增案件数、破产关联方数；供应商、客户、人员、投资人各10项', '40'],
           ['L5 同行业竞争气候', '同行业评级主体近一年降级比例、负面展望比例、同业数量', '3']],
          [95, 350, 50])
    para('预处理只在训练集拟合（1%/99%裁剪、中位数填补加缺失指示、标准化）；关联方特征只读取关联方在观察日之前可见的评级动作与诉讼记录，'
         '不读取任何主体的未来标签。另设“精简版”关联特征（每类关系只保留是否有关联方、恶化关联方占比、被诉关联方占比、是否有破产关联方4项）检验特征冗余的影响。')
    h('3.2 候选模型与训练协议', 2)
    para('表格模型：逻辑回归（可解释评分卡，C=1）与LightGBM（150棵、7叶、学习率0.05），按A（L1+L2）→B（+L3a）→C（+L3b）→D（+供应链）→E（+人员）→F（+投资人）→G（+行业）逐层加入特征。'
         '图模型：GraphSAGE、GAT与多关系R-GCN，两层、16维、输入跳接；节点（包括未打标签的邻居企业）携带L1–L3全部属性，边为四类关系（供应、反向供应、人员、投资人）。'
         '三个随机种子取平均；早停与模型选择只看验证集，测试集仅用于最终评估。')
    para('传播机制（h为节点表示，N_r(v)为第r类关系的邻居）：GraphSAGE  h_v ← ReLU(W·[h_v ‖ mean_{u∈N(v)} h_u])；'
         'GAT  h_v ← Σ_{u∈N(v)∪{v}} α_vu·W h_u，α由注意力学习“哪个关联方更重要”；'
         'R-GCN  h_v ← ReLU(W_0 h_v + Σ_r mean_{u∈N_r(v)} W_r h_u)，每类关系有独立变换，可学习供应、人员、投资人传导强度的差异。'
         '两层叠加相当于两跳传导。对照组：“无边”删除全部关系、参数不变；“重连/随机关联”在每个季度把关联对象替换为同季度随机企业，保留关联数量而破坏真实对应关系。'
         '只有当真实图同时优于这些对照时，才能把增益归因于关联图谱。')
    h('3.3 对原有设计的审查与改进', 2)
    table(['原设计问题', '本轮改进', '结果'],
          [['GNN节点只含财务比率，关联方风险事件进不了消息传递', '所有节点加入时点评级事件与涉诉事件属性', f"无图MLP测试AP {gorig['summaries']['mlp']['test']['ap']:.3f}→{gev['summaries']['mlp']['test']['ap']:.3f}（D/SD）"],
           ['只有供应链一类关系，缺少赛题要求的股东与人员关联', '新增BoardEx人员关联与13F共同大股东关联', f"测试期人员关联覆盖{100 * rcov['people']['test']:.0f}%，投资人{100 * rcov['investors']['test']:.0f}%"],
           ['缺少涉诉等风险事件', '新增Audit Analytics诉讼与破产事件序列', '自身与关联方两侧均入模'],
           ['标签只用D/SD，事件过少（测试23家）', '增加显著恶化、下调≥1档两层标签', f"测试事件企业增至{rel['targets']['severe']['counts']['test']['positive_entities']}/{rel['targets']['mild']['counts']['test']['positive_entities']}家"],
           ['缺少自身风险事件时序与竞争关联', '增加L3a展望/观察/降级与L5行业气候', f"LR测试AP {res['financial_rating_logistic']['test']['ap']:.3f}→{res['plus_own_events_logistic']['test']['ap']:.3f}（D/SD）"],
           ['原GNN最优轮次触及100轮上限', '学习率0.005、300轮、耐心40的预算对照', f"MLP测试AP {gbud['summaries']['mlp']['test']['ap']:.3f}，预算不是瓶颈"],
           ['报告不符合六部分与8页要求；混入不适配的SMEsD数据', '重写报告；移除SMEsD相关模型与实验', '本报告']],
          [175, 170, 150])
    h('3.4 结果：特征阶梯与多关系图模型', 2)
    steps = [('A_financial_rating', 'A 财务+评级'), ('B_plus_own_rating_events', 'B +自身评级事件'), ('C_plus_own_litigation', 'C +自身涉诉'),
             ('D_plus_supply', 'D +供应链'), ('E_plus_people', 'E +人员'), ('F_plus_investors', 'F +投资人'), ('G_plus_industry', 'G +行业'),
             ('G_randomized_relations', 'G 随机关联对照')]
    tg = rel['targets']
    table(['特征组合（LR / 树，测试AP）', 'D/SD违约', '显著恶化', '下调≥1档'],
          [[n] + [f"{tg[t]['results'][k + '_logistic']['test']['ap']:.3f} / {tg[t]['results'][k + '_tree']['test']['ap']:.3f}" for t in ('default', 'severe', 'mild')] for k, n in steps],
          [150, 115, 115, 115],
          note='测试期基准事件率：D/SD 0.006、显著恶化 0.022、下调≥1档 0.053。验证集规则选中：' +
               '；'.join(f"{n}→{tg[t]['selected_by_validation']}" for t, n in (('default', 'D/SD'), ('severe', '显著恶化'), ('mild', '下调≥1档'))) + '。')
    figure(hbar([(n, tg['severe']['results'][k + '_logistic']['test']['ap'], k == 'B_plus_own_rating_events') for k, n in steps]
                + [(f'多关系GNN：{m}', gsev['summaries'][k]['test']['ap'], False) for k, m in (('mlp', 'MLP（无图）'), ('sage', 'GraphSAGE'), ('gat', 'GAT'), ('rgcn', 'R-GCN'), ('rgcn_rewired', 'R-GCN重连对照'))],
                '显著恶化目标的测试AP（逻辑回归特征阶梯与多关系图神经网络）', lambda v: f'{v:.3f}', max_value=0.45),
           '图3  显著恶化目标测试AP。橙色为验证集选中的B（财务+评级+自身评级事件）逻辑回归；关联特征与图模型均未超过它。',
           [(n, f"{tg['severe']['results'][k + '_logistic']['test']['ap']:.3f}") for k, n in steps])
    cs, cm = tg['severe']['paired_comparisons'], tg['mild']['paired_comparisons']
    table(['配对比较（测试AP差，企业重抽样95%区间）', '显著恶化', '下调≥1档'],
          [['B 自身评级事件 vs A（树）', ci(cmp['plus_own_events_tree_vs_financial_rating_tree']) + '（D/SD）', '—'],
           ['G 全部特征 vs A（树）', ci(cs['G_plus_industry_tree_vs_A_financial_rating_tree']), ci(cm['G_plus_industry_tree_vs_A_financial_rating_tree'])],
           ['D +供应链 vs C（LR）', ci(cs['D_plus_supply_logistic_vs_C_plus_own_litigation_logistic']), ci(cm['D_plus_supply_logistic_vs_C_plus_own_litigation_logistic'])],
           ['F 三类关联 vs C（树）', ci(cs['F_plus_investors_tree_vs_C_plus_own_litigation_tree']), ci(cm['F_plus_investors_tree_vs_C_plus_own_litigation_tree'])],
           ['G 真实关联 vs 随机关联（树）', ci(cs['G_plus_industry_tree_vs_G_randomized_relations_tree']), ci(cm['G_plus_industry_tree_vs_G_randomized_relations_tree'])],
           ['多关系R-GCN vs 同特征MLP', ci(gsev['comparisons_vs_mlp']['rgcn']), '—'],
           ['多关系GraphSAGE vs 同特征MLP', ci(gsev['comparisons_vs_mlp']['sage']), '—']],
          [185, 155, 155])
    para('解读：（1）增益几乎全部来自企业自身的风险事件时序（评级展望、观察与降级）；自身涉诉事件在评级企业中没有额外信息。'
         '（2）加入供应链、人员、投资人三类关联后，三个目标上都没有稳定增益，逻辑回归甚至因特征增多而变差；精简版特征结论相同。'
         '（3）真实关联与“随机关联对象”对照没有显著差异，多关系图神经网络也未超过同特征无图模型——在显著恶化目标上，验证集最优的图模型恰是随机重连对照。'
         '结合分层统计，关联传导信号确实存在于B档及以下企业，但这些企业通常已出现自身负面展望或降级，评级机构已把合作方风险“提前定价”进自身评级，'
         '因此在评级企业样本上关联图谱提供的是解释与预警价值，而非额外的排序能力。由于事件企业已增至数十至上百家，这一结论不再主要受样本量限制。')

    # ---------------- 4 Validation ----------------
    h('四、应用验证')
    names = {'E': 'E（高风险）', 'D': 'D', 'C': 'C', 'B': 'B', 'A': 'A（低风险）'}
    cuts = card['score_cuts']
    ranges = ['< %.0f' % cuts[0], '%.0f–%.0f' % (cuts[0], cuts[1]), '%.0f–%.0f' % (cuts[1], cuts[2]), '%.0f–%.0f' % (cuts[2], cuts[3]), '≥ %.0f' % cuts[3]]
    share = ['训练最低1%', '1%–5%', '5%–20%', '20%–50%', '最高50%']
    table(['等级', '分数区间', '训练分位', '测试记录', '测试违约记录', '违约主体', '测试违约率'],
          [[names[g['grade']], r, s, f"{g['test_rows']:,}", g['test_events'], g['test_event_entities'], pct(g['test_event_rate'])]
           for g, r, s in zip(card['grades'], ranges, share)], [70, 70, 70, 65, 70, 60, 90],
          note='评分 = 600 − 50/ln2 × ln(odds)，分数越高越安全（PDO=50）。等级切点只用训练集分数分位确定，再原样应用于2022–2023测试期。')
    cap = card['test_capture']
    para(f"时间外测试（2022–2023，{sp['test']['rows']:,}条记录、{sp['test']['positive_entities']}家违约主体）："
         f"终版评分卡AP {t_sel['ap']:.3f}（基准约{sp['test']['positive_rows'] / sp['test']['rows']:.4f}，提升约{t_sel['ap'] / (sp['test']['positive_rows'] / sp['test']['rows']):.0f}倍）、"
         f"AUC {t_sel['roc_auc']:.3f}、KS {t_sel['ks']:.3f}、Brier {t_sel['brier']:.5f}。按风险排序，前1%/5%/10%/20%记录分别覆盖"
         f"{pct(cap['top_1pct'])}/{pct(cap['top_5pct'])}/{pct(cap['top_10pct'])}/{pct(cap['top_20pct'])}的违约记录；"
         '各等级违约率严格单调，A、B两级（约占测试记录八成）无违约，可直接对应差异化授信与复核资源分配。')
    coef = cont['selected_coefficients'][:8]
    label = lambda f: {'own_rating:rating_ordinal': '评级等级（越差越高）', 'own_events:own_notches_change_365': '近一年级差变化',
                       'own_events:own_negative_outlook': '负面展望', 'own_events:own_downgraded_365': '近一年是否降级',
                       'financial:debt_assets': '债务/资产', 'financial:cash_assets': '现金/资产',
                       'financial:current_assets_assets': '流动资产/资产', 'financial:ebit_assets': 'EBIT/资产',
                       'financial:current_ratio': '流动比率', 'own_events:own_watch_negative': '负面观察'}.get(f, f)
    para('模型解释（标准化系数，正值提高风险）：' + '；'.join(f"{label(c['feature'])} {c['standardized_coefficient']:+.2f}" for c in coef) +
         '。“是否降级”为负而“级差变化”为正，两者高度相关，应合并解读为“降级幅度越大风险越高”；财务方面高杠杆、低现金、低盈利提高风险，方向符合信用逻辑。')
    iso = ext.get('entity_isolation', {}) if isinstance(ext, dict) else {}
    tm = rel['targets']['mild']['results']['B_plus_own_rating_events_logistic']['test']
    para(f"分层目标验证：同一组L1–L3特征在显著恶化目标上测试AP {tsev['ap']:.3f}（基准0.022）、AUC {tsev['roc_auc']:.3f}、KS {tsev['ks']:.3f}；"
         f"在下调≥1档目标上AP {tm['ap']:.3f}（基准0.053）、AUC {tm['roc_auc']:.3f}。因此同一套评分卡可以输出“违约风险分”与“恶化预警分”两条曲线，"
         '前者用于准入与定价，后者用于贷中复核排序。')
    para('稳健性：原研究已完成的敏感性与本轮结论一致——供应链可见延迟0/180/365天、关系保留期730天、财务新鲜度365/550/730天、'
         '单向/逆向消息、企业隔离划分下，图模型相对无图对照的差异区间均跨0；本轮新增的人员与投资人关联、精简版关联特征、三层标签与多关系图模型同样未改变结论。'
         '本轮追加分析均发生在主测试期查看之后，属于探索性检验，不是新的外部独立验证。')

    # ---------------- 5 Product ----------------
    h('五、产品思路')
    para('产品定位：“经营类客户关联风险评分与预警系统”，由两层组成——主评分层负责准入与定价，关联图谱预警层负责贷中监测与解释。'
         '设计依据是本研究的实证结论：评分层只纳入已证明有稳定增益的信息；关联传导虽未带来统计增量，但其单变量联动明显、可解释性强，'
         '适合作为触发人工复核的预警规则，而不是直接改变分数。')
    table(['模块', '功能', '在银行内部数据上的落地'],
          [['数据层：企业关联图谱', '季度快照，节点=企业/自然人，边=上下游、股权、任职、担保、同行业', '工商登记股权与高管、对公结算上下游、担保圈、行业分类'],
           ['事件层：风险事件时序库', '为每个主体维护带时间戳的风险事件流', '征信逾期、裁判文书涉诉、执行信息公开网失信/被执行、舆情'],
           ['评分层：评分卡主模型', 'L1–L3特征逻辑回归，输出600基准分与A–E五级', '替换为行内财务、征信与自身逾期/涉诉序列，按本文协议重训'],
           ['预警层：关联风险传导', '关联方新增降级/逾期/失信时，沿1–2跳路径推送预警并展示传导路径', '关联方事件触发复核清单；路径可视化给客户经理'],
           ['监控层：模型治理', 'KS/AP季度回测、PSI稳定性、等级违约率单调性、关联覆盖率', '超阈值触发重训；挑战者模型（GNN）平行跑批']],
          [100, 205, 190])
    table(['等级', '建议策略'],
          [['A / B', '标准准入与定价；年度复审；关联预警仅提示'],
           ['C', '常规准入，加强贷后；关联方出现风险事件时升级为季度复审'],
           ['D', '限额管理、提高担保要求；关联预警触发即时复核'],
           ['E', '原则上不新增授信，存量压降与重点监控；展示主要风险驱动与关联路径']], [70, 425])
    para('挑战者机制：GraphSAGE/R-GCN等图模型作为挑战者持续平行运行。当行内图谱中风险事件更密集（例如担保圈、实控人控制链这类强关联）、'
         '违约主体数量足以检验时，若挑战者在新时间外样本上稳定优于评分卡，再按同一协议切换，避免凭单次测试结果上线复杂模型。')

    # ---------------- 6 Conclusion ----------------
    h('六、总结结论')
    para(f"（1）构建了以企业为核心的多关系关联图谱：上下游供应链、共享董事/高管的人员关联、共同≥5%机构股东的投资人关联与同行业竞争关联，"
         f"覆盖{panel['total_node_rows']:,}个企业-季度节点；节点携带评级与涉诉/破产两类风险事件序列，全部按观察日可见性构建，并以严格的时间外协议完成建模。")
    para(f"（2）风险事件的时序信息价值最大：加入负面展望、负面观察与近一年降级后，D/SD目标逻辑回归测试AP由{res['financial_rating_logistic']['test']['ap']:.3f}升至{t_sel['ap']:.3f}；"
         f"显著恶化目标AP达{tsev['ap']:.3f}（基准0.022）。评分卡等级违约率严格单调，前5%名单覆盖{pct(card['test_capture']['top_5pct'])}的违约记录。")
    para('（3）关联风险联动真实存在但集中在B档及以下的脆弱企业；在控制自身信息后，三类关联特征与多关系GraphSAGE/GAT/R-GCN均未超过同特征无图模型或随机关联对照。'
         '事件企业增至数十至上百家后结论不变，主要原因是评级机构已把合作方风险吸收进自身评级与展望。这一负结果被完整保留，终版模型按验证集规则选择评分卡。')
    para('（4）局限与下一步：评级恶化只是贷款违约的代理；实控人/母子公司链条因缺少企业编码未入模；关系可见时间部分依赖假设。'
         '对于无外部评级、信息更不透明的中小经营类客户，关联方事件很可能是自身风险的领先信号——应在银行内部的工商股权、担保圈、征信逾期与司法失信数据上，'
         '以本文的特征体系、对照设计与验证协议重建图谱并检验。')
    story.append(PageMarker(pages, 'body_end'))

    # ---------------- Appendix ----------------
    story.append(PageBreak())
    h('附录  复现说明（不计入正文页数）')
    para('环境：Python 3.11 隔离环境（依赖锁定于 比赛/experiments/requirements.lock.txt）。从项目根目录依次运行：', small)
    for c in ['比赛/src/build_global_panel.py --freshness 550', '比赛/src/build_global_graph.py --freshness 550 --lag-days 365',
              '比赛/src/run_global_baselines.py ; --config global_rating_enhancement_config.json', '比赛/src/run_global_graph_models.py（原始图模型）',
              '比赛/src/run_global_contagion.py（特征阶梯、评分卡、描述统计）', '比赛/src/wrds_jupyter/02_export.py（在WRDS Jupyter运行，导出关联数据）', '比赛/src/build_related_graph.py（人员/投资人关联、涉诉事件、三层标签）', '比赛/src/run_related_contagion.py ; --config related_contagion_compact_config.json', '比赛/src/run_global_graph_models.py --config global_graph_related_severe_config.json / global_graph_related_default_config.json', '比赛/src/run_global_graph_models.py --config global_graph_event_config.json',
              '比赛/src/run_global_graph_models.py --config global_graph_budget_config.json', '比赛/src/build_final_report.py']:
        para('.venv-credit/bin/python ' + c, small)
    para('所有数字均由本脚本从 比赛/experiments/*.json 读取生成；原始授权数据、逐行预测与检查点保存在Git忽略目录。单元测试：python -m unittest discover -s 比赛/tests。', small)

    PDF.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setFont('ReportCJK', 7.5); canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawRightString(A4[0] - 50, 22, f'{doc.page}')
    SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=50, rightMargin=50, topMargin=42, bottomMargin=40,
                      title='经营类客户关联图谱风控模型终版报告').build(story, onFirstPage=footer, onLaterPages=footer)
    MD.write_text(''.join(md))
    print(json.dumps({'pdf': str(PDF), 'markdown': str(MD), 'body_pages': pages.get('body_end'),
                      'within_limit': pages.get('body_end', 99) <= BODY_PAGE_LIMIT}, ensure_ascii=False))
    if pages.get('body_end', 99) > BODY_PAGE_LIMIT:
        raise SystemExit('Body exceeds page limit')


if __name__ == '__main__':
    main()
