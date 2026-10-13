"""Submission report for the competition (six required sections, body <= 8 pages), business-facing, with vector figures.

Main model: non-backtracking GraphSAGE (sage_gnn.py), compared with common industry baselines. Every claim is checked
against the current experiment aggregates, including the ones that do not favour the graph model (random-relation
control, debt).

Narrative order: section 4 holds everything measured on the test period (early warning, grades, explainability,
measured debt); section 5 holds product use, the input/output estimate built on assumed parameters, and the roll-out
path. The previous layout is kept as reports/备份/参赛报告_原版*.

All numbers are read from saved experiment aggregates (experiments/*.json). Business value is led by the measured debt
of the distressed firms each model flags early (experiments/business_value.json, measured_debt); the conversion into
avoided loss uses the typed LGD and loss-reduction parameters stored next to it.
"""
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Circle, Drawing, Group, Line, PolyLine, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import CondPageBreak, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from build_final_report_rated import FONT, PageMarker

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
PDF = ROOT / 'reports/参赛报告_经营类客户关联图谱风控模型.pdf'
MD = ROOT / 'reports/参赛报告.md'
BODY_PAGE_LIMIT = 8
REPO_URL = 'https://github.com/wangzhihao-coder-dev/kg-credit-risk-competition'
INK, INK2, MUTED, GRID = '#1f2a37', '#52514e', '#898781', '#e4e3df'
BLUE, ORANGE, AQUA, YELLOW, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#d03b3b'
NAVY = '#13334e'
W = 495  # text width


def load(name):
    return json.loads((EXP / name).read_text())


def txt(d, x, y, s, size=8, color=INK, anchor='start'):
    d.add(String(x, y, s, fontName='ReportCJK', fontSize=size, fillColor=colors.HexColor(color), textAnchor=anchor))


def box(d, x, y, w, h, title, lines=(), fill='#eef4fc', stroke=BLUE, title_size=9):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor(stroke), strokeWidth=1))
    txt(d, x + w / 2, y + h - 13, title, title_size, INK, 'middle')
    for i, line in enumerate(lines):
        txt(d, x + w / 2, y + h - 25 - 10 * i, line, 7, INK2, 'middle')


def arrow(d, x1, y1, x2, y2, color=MUTED):
    d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor(color), strokeWidth=1.2))
    ang = math.atan2(y2 - y1, x2 - x1); a, s = 0.45, 6
    d.add(Polygon([x2, y2, x2 - s * math.cos(ang - a), y2 - s * math.sin(ang - a), x2 - s * math.cos(ang + a), y2 - s * math.sin(ang + a)],
                  fillColor=colors.HexColor(color), strokeColor=None))


def framework():
    d = Drawing(W, 118)
    steps = [('多源数据', ['财务 / 股价 / 评级', '涉诉 / 延迟披露 / 破产', '供应链 / 任职 / 持股']),
             ('企业关联图谱', ['企业节点·季度快照', '供应·人员·股东三类边', '风险事件时序属性']),
             ('关联风险传导模型', ['客户编码器+关联方编码器', 'GraphSAGE 均值聚合', '非回溯两跳传导']),
             ('风险评分与解释', ['600分基准 A–E 五级', '传导路径归因', '季度更新']),
             ('业务处置', ['准入与定价', '贷中预警与复核', '限额与压降'])]
    bw, gap = 88, 13
    for i, (t, ls) in enumerate(steps):
        x = i * (bw + gap)
        box(d, x, 22, bw, 78, t, ls, fill='#fff4ee' if i in (2, 4) else '#eef4fc', stroke=ORANGE if i in (2, 4) else BLUE)
        if i:
            arrow(d, x - gap + 1, 61, x - 2, 61)
    txt(d, W / 2, 6, '客户只需提供行内可得信息；关联方信息来自公开数据与行内关系数据', 7.5, INK2, 'middle')
    return d


def graph_schema():
    d = Drawing(W, 150)
    box(d, 178, 90, 140, 52, '目标经营类客户', ['只用：报表与风险事件', '（无股价、无外部评级）'], '#fff4ee', ORANGE)
    box(d, 5, 90, 125, 52, '上游供应商', ['股价·评级·事件', '供货关系'])
    box(d, 365, 90, 125, 52, '下游客户', ['股价·评级·事件', '收入依赖度（边权）'])
    box(d, 5, 6, 150, 52, '人员关联企业', ['同一董事/高管同时在任', '常对应同一集团/实控人'])
    box(d, 172, 6, 150, 52, '投资人关联企业', ['同一机构持股均 ≥5%', '共同大股东'])
    box(d, 340, 6, 150, 52, '两跳关联方', ['关联方的关联方', '两层传导，不回传客户自身'])
    for x1, y1, x2, y2, t in [(130, 116, 178, 116, '供货'), (318, 116, 365, 116, '供货'), (210, 90, 80, 58, '共享人员'),
                              (248, 90, 248, 58, '共同股东'), (286, 90, 415, 58, '多跳')]:
        d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor(MUTED), strokeWidth=1))
        txt(d, (x1 + x2) / 2 + 4, (y1 + y2) / 2 + 3, t, 7, INK2, 'middle')
    return d


def architecture():
    d = Drawing(W, 196)
    box(d, 0, 124, 108, 46, '客户自身特征', ['财务比率·规模', '涉诉·延迟披露·重述'], '#fff4ee', ORANGE)
    box(d, 0, 44, 108, 46, '关联方完整特征', ['+股价·评级', '（上市关联方公开可得）'])
    box(d, 126, 124, 78, 46, '客户编码器', ['线性+ReLU'], '#fff4ee', ORANGE)
    box(d, 126, 44, 78, 46, '关联方编码器', ['线性+ReLU'])
    arrow(d, 108, 147, 124, 147); arrow(d, 108, 67, 124, 67)
    box(d, 224, 36, 120, 62, '第1层：关联方 j', ['对 j 的关联方取均值', '（不含客户 i 本身）', '+ j 自身表示'])
    box(d, 224, 116, 120, 62, '第2层：客户 i', ['对各关联方第1层', '表示取均值', '+ 客户自身表示'], '#fff4ee', ORANGE)
    arrow(d, 204, 67, 222, 67); arrow(d, 204, 147, 222, 147); arrow(d, 284, 98, 284, 114)
    box(d, 368, 116, 122, 62, '风险评分', ['拼接客户原始输入', 'P(12个月内困境)', '→ 600分 A–E 五级'], '#fff4ee', ORANGE)
    arrow(d, 344, 147, 366, 147)
    box(d, 368, 30, 122, 72, '非回溯约束', ['j 发给 i 的消息先剔除', 'i 自己的信息，客户自身', '股价不会经关联方', '“回声”回到自己'], '#fdecec', RED)
    d.add(Line(344, 67, 366, 67, strokeColor=colors.HexColor(RED), strokeWidth=1, strokeDashArray=[2, 2]))
    txt(d, 0, 10, '供应、人员、股东三类关系合并为一张关联网（GraphSAGE 均值聚合）；正样本按 √(负/正) 加权；早停只看验证期；三个随机种子平均。', 7.2, INK2)
    return d


def vbars(items, title, fmt, height=150, max_value=None, ylabel=''):
    """items: (label, value, color)."""
    d = Drawing(W, height); max_value = max_value or max(v for _, v, _ in items) * 1.2
    left, bottom, top = 34, 30, height - 20; plot_h = top - bottom; n = len(items); slot = (W - left - 5) / n; bw = min(34, slot * 0.6)
    txt(d, 0, height - 10, title, 8.5, INK2)
    for k in range(5):
        y = bottom + plot_h * k / 4
        d.add(Line(left, y, W, y, strokeColor=colors.HexColor(GRID), strokeWidth=.5)); txt(d, left - 4, y - 3, fmt(max_value * k / 4), 6.5, MUTED, 'end')
    for i, (lab, v, c) in enumerate(items):
        x = left + slot * i + (slot - bw) / 2; h = plot_h * v / max_value
        d.add(Rect(x, bottom, bw, max(h, 1), fillColor=colors.HexColor(c), strokeColor=None))
        txt(d, x + bw / 2, bottom + h + 3, fmt(v), 7.2, INK, 'middle')
        for j, part in enumerate(lab.split('\n')):
            txt(d, x + bw / 2, bottom - 10 - 9 * j, part, 7, INK2, 'middle')
    return d


def gains_chart(shares, series, height=190):
    d = Drawing(W, height); left, bottom, right, top = 40, 34, W - 120, height - 14
    xs = lambda v: left + (right - left) * v / 0.30; ys = lambda v: bottom + (top - bottom) * v
    for k in range(6):
        y = k / 5; d.add(Line(left, ys(y), right, ys(y), strokeColor=colors.HexColor(GRID), strokeWidth=.5)); txt(d, left - 4, ys(y) - 3, f'{int(y * 100)}%', 6.5, MUTED, 'end')
    for x in (0, .05, .10, .15, .20, .25, .30):
        txt(d, xs(x), bottom - 11, f'{int(x * 100)}%', 6.5, MUTED, 'middle')
    txt(d, (left + right) / 2, bottom - 24, '每季度复核的客户比例（按模型风险分从高到低）', 7.5, INK2, 'middle')
    txt(d, 0, height - 8, '抓到的未来12个月困境事件占比', 7.5, INK2)
    d.add(Line(xs(.05), bottom, xs(.05), top, strokeColor=colors.HexColor(MUTED), strokeWidth=.6, strokeDashArray=[2, 2]))
    for i, (name, ys_, color, dash) in enumerate(series):
        pts = [xs(0), ys(0)] + sum(([xs(s), ys(g)] for s, g in zip(shares, ys_)), [])
        d.add(PolyLine(pts, strokeColor=colors.HexColor(color), strokeWidth=2, strokeDashArray=dash))
        ly = top - 14 * i - 6
        d.add(Line(right + 10, ly + 3, right + 26, ly + 3, strokeColor=colors.HexColor(color), strokeWidth=2, strokeDashArray=dash)); txt(d, right + 30, ly, name, 7.5, INK)
    return d


def roc_grid(roc, panels, reference, cols=3, height=268):
    """One ROC panel per model (panels: (key, title, color)); the reference model is repeated faintly in every other panel."""
    d = Drawing(W, height); rows = math.ceil(len(panels) / cols); cell_w, cell_h = W / cols, height / rows
    fpr = roc['fpr']; ref = roc['models'][reference]['tpr']
    for i, (k, title, color) in enumerate(panels):
        ox, oy = cell_w * (i % cols), height - cell_h * (i // cols + 1)
        left, bottom = ox + 30, oy + 26; size = min(cell_w - 42, cell_h - 46)
        xs = lambda v: left + size * v; ys = lambda v: bottom + size * v
        for v in (0, .5, 1):
            d.add(Line(left, ys(v), xs(1), ys(v), strokeColor=colors.HexColor(GRID), strokeWidth=.5))
            d.add(Line(xs(v), bottom, xs(v), ys(1), strokeColor=colors.HexColor(GRID), strokeWidth=.5))
            txt(d, left - 3, ys(v) - 2.5, f'{v:g}', 6, MUTED, 'end'); txt(d, xs(v), bottom - 8, f'{v:g}', 6, MUTED, 'middle')
        d.add(Line(xs(0), ys(0), xs(1), ys(1), strokeColor=colors.HexColor(MUTED), strokeWidth=.6, strokeDashArray=[2, 2]))
        if k != reference:
            d.add(PolyLine(sum(([xs(x), ys(t)] for x, t in zip(fpr, ref)), []), strokeColor=colors.HexColor('#f5c2ab'), strokeWidth=1, strokeDashArray=[3, 2]))
        d.add(PolyLine(sum(([xs(x), ys(t)] for x, t in zip(fpr, roc['models'][k]['tpr'])), []), strokeColor=colors.HexColor(color), strokeWidth=1.8))
        txt(d, left, ys(1) + 7, title, 7.5, INK)
        txt(d, xs(1) - 3, bottom + 5, f"AUC {roc['models'][k]['auc']:.3f}", 8, color, 'end')
        txt(d, xs(.5), bottom - 17, '误报率 FPR', 6.3, INK2, 'middle')
        label = Group(String(0, 0, '命中率 TPR', fontName='ReportCJK', fontSize=6.3, fillColor=colors.HexColor(INK2), textAnchor='middle'))
        label.transform = (0, 1, -1, 0, left - 16, ys(.5)); d.add(label)
    return d


def ego_case(case, height=230):
    d = Drawing(W, height); cx, cy, r = 140, height / 2 + 2, 66
    names = {'supplier': '供应商', 'customer': '客户', 'people': '共享董事/高管', 'investor': '共同大股东'}
    nbrs = case['neighbours'][:8]
    for i, n in enumerate(nbrs):
        a = math.pi / 2 + 2 * math.pi * (i + 0.5) / len(nbrs); ca, sa = math.cos(a), math.sin(a)
        x, y = cx + r * ca, cy + r * sa
        hot = n['risk_p'] > 0.2 or n['recent_distress']
        d.add(Line(cx, cy, x, y, strokeColor=colors.HexColor(ORANGE if hot else '#c8d3de'), strokeWidth=2.2 if hot else 1))
        radius = 8 + 10 * min(n['risk_p'], 0.6)
        d.add(Circle(x, y, radius, fillColor=colors.HexColor(RED if hot else '#d9e8fb'), strokeColor=colors.white))
        sig = ([f"股价{n['ret_12m'] * 100:.0f}%"] if n['ret_12m'] is not None and n['ret_12m'] < -0.5 else []) + \
              (['延迟披露'] if n['late_filing'] else []) + (['已陷困境'] if n['recent_distress'] else [])
        lx, ly = cx + (r + radius + 8) * ca, cy + (r + radius + 8) * sa
        anchor = 'start' if ca > 0.25 else ('end' if ca < -0.25 else 'middle')
        top = ly + (4 if sa > 0.25 else (-8 if sa < -0.25 else 2))
        txt(d, lx, top, names[n['relation']], 7, INK, anchor)
        if sig:
            txt(d, lx, top - 9, '·'.join(sig), 6.6, RED, anchor)
    d.add(Circle(cx, cy, 20, fillColor=colors.HexColor(ORANGE), strokeColor=colors.white, strokeWidth=2)); txt(d, cx, cy - 3, '客户', 8, '#ffffff', 'middle')
    x0 = 305
    txt(d, x0, height - 22, '案例：自身报表风险不突出的客户', 9, NAVY)
    names_short = {'supplier': '供应商', 'customer': '客户', 'people': '共享董事企业', 'investor': '共同大股东企业'}
    signals = []
    for n in case['neighbours']:
        parts = ([f"股价跌{-n['ret_12m'] * 100:.0f}%"] if n['ret_12m'] is not None and n['ret_12m'] < -0.5 else []) + \
                (['延迟披露'] if n['late_filing'] else []) + (['已陷困境'] if n['recent_distress'] else [])
        if parts:
            signals.append(names_short[n['relation']] + '：' + '、'.join(parts))
    quarter = {'03': '一', '06': '二', '09': '三', '12': '四'}[case['quarter'][5:7]]
    rows = [f"观察时点：{case['quarter'][:4]}年{quarter}季度末；共 {case['neighbour_count']} 个关联方",
            f"不用图谱：同季风险排名第 {case['own_rank_pct'] * 100:.0f} 百分位（未进前25%）",
            f"关联图谱模型：同季排名前 {case['graph_rank_pct'] * 100:.1f}%，进入复核名单"] + \
           [('关联信号：' if i == 0 else '　　　　　') + sig for i, sig in enumerate(signals[:3])] + \
           ['结果：该客户在随后12个月内陷入困境（破产/摘牌）']
    for i, row in enumerate(rows):
        txt(d, x0, height - 42 - 16 * i, row, 7.6, RED if i == len(rows) - 1 else INK)
    txt(d, x0, 12, '红色节点 = 高风险关联方（节点越大风险越高）；企业已匿名', 6.8, MUTED)
    return d


def main():
    U = load('distress_distress_unlisted_results.json'); UB = load('distress_bankruptcy_unlisted_results.json'); F = load('distress_distress_results.json')
    S = load('distress_distress_unlisted_gnn_study.json'); B = load('business_value.json'); A = load('distress_panel_audit.json')
    main_run = S['runs']['main_all_relations_full_neighbour_view']['test']; base = B['base_rate']
    gains, shares = B['gains'], B['review_shares']; g = lambda k, s: gains[k][shares.index(s)]
    prec = lambda k, s: g(k, s) * base / s
    D = B['measured_debt']; dm = D['by_model']; conc = D['concentration']
    snaps = A['snapshots']; mean_edges = lambda k: sum(v[k] for v in snaps.values()) / len(snaps)
    run = lambda k, s='test': U['runs'][k][s]; cmp_ = lambda a, b: U['comparisons'][f'{a}_vs_{b}']

    pdfmetrics.registerFont(TTFont('ReportCJK', FONT))
    body = ParagraphStyle('body', fontName='ReportCJK', fontSize=9.4, leading=14.6, spaceAfter=5, wordWrap='CJK', textColor=colors.HexColor('#26354a'))
    title_s = ParagraphStyle('title', parent=body, fontSize=18, leading=25, spaceAfter=3, textColor=colors.HexColor(NAVY))
    h1 = ParagraphStyle('h1', parent=body, fontSize=12.8, leading=18, spaceBefore=7, spaceAfter=5, textColor=colors.HexColor(NAVY))
    h2 = ParagraphStyle('h2', parent=body, fontSize=10.2, leading=15, spaceBefore=3, spaceAfter=3, textColor=colors.HexColor(NAVY))
    small = ParagraphStyle('small', parent=body, fontSize=8, leading=11.5, spaceAfter=4, textColor=colors.HexColor(INK2))
    key = ParagraphStyle('key', parent=body, fontSize=9.4, leading=14.6, backColor=colors.HexColor('#f3f7fb'), borderPadding=(6, 8, 6, 8),
                         spaceBefore=4, spaceAfter=9, textColor=colors.HexColor('#1f2a37'))
    cell = ParagraphStyle('cell', parent=small, fontSize=8, leading=11, spaceAfter=0, textColor=colors.HexColor(INK))
    story, md, pages = [], [], {}

    def para(text, style=body, prefix=''):
        story.append(Paragraph(escape(text), style)); md.append(prefix + text + '\n\n')
    def h(text, level=1):
        story.append(CondPageBreak(110))  # never leave a heading stranded at the foot of a page
        para(text, h1 if level == 1 else h2, '## ' if level == 1 else '### ')
    def table(headers, rows, widths, highlight=None, note=None):
        data = [[Paragraph(escape(str(v)), cell) for v in row] for row in [headers] + rows]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        st = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e7eff6')), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
              ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4), ('TOPPADDING', (0, 0), (-1, -1), 2.8),
              ('BOTTOMPADDING', (0, 0), (-1, -1), 2.8), ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#8ba6bf')),
              ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')])]
        if highlight is not None:
            st.append(('BACKGROUND', (0, highlight + 1), (-1, highlight + 1), colors.HexColor('#fff4ee')))
        t.setStyle(TableStyle(st)); story.extend([t, Spacer(1, 4)])
        md.append('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n'); md.extend('| ' + ' | '.join(map(str, r)) + ' |\n' for r in rows); md.append('\n')
        if note:
            para(note, small)
    def figure(drawing, caption):
        story.append(KeepTogether([drawing, Spacer(1, 2), Paragraph(escape(caption), small)])); md.append(f'**{caption}**\n\n')
    pct = lambda x, n=1: f'{100 * x:.{n}f}%'
    yi = lambda m, n=0: f'{m / 100:,.{n}f}亿美元'  # source unit: millions
    ci = lambda pair: f"{pair[0] / 100:,.0f}～{pair[1] / 100:,.0f}亿美元"
    rng = lambda c, k='ap': f"{c[k + '_difference']:+.3f}（95%区间{c[k + '_difference_ci95'][0]:+.3f}～{c[k + '_difference_ci95'][1]:+.3f}）"
    half = B['review_share_for_half_of_events']
    firms_gap = dm['graph']['firms'] - dm['own_scorecard']['firms']; firms_ci = conc['graph_minus_model_firms_ci95']['own_scorecard']
    vs_mlp, vs_card, vs_rand = cmp_('sage_nb', 'mlp'), cmp_('sage_nb', 'gbdt_own_unlisted'), cmp_('sage_nb', 'sage_nb_rewired')

    # ---------------- Cover + abstract ----------------
    para('基于关联图谱的经营类客户风险预警模型', title_s, '# ')
    para('让“看不见”的关联风险进入授信决策：上下游、董事高管与大股东关联中的风险传导识别', small)
    para("经营类客户的风险常常不是从自身报表开始暴露的：上游供应商断供、共享董事的关联企业爆雷、大股东旗下企业接连出事，往往先于客户自身财务恶化。"
         "本项目构建覆盖供应链、董事高管任职与机构持股三类关系、并携带涉诉、延迟披露、财务重述、破产等风险事件时序的企业关联图谱，"
         "提出“客户编码器+关联方编码器+非回溯GraphSAGE”的图神经网络：关联方的风险经两层传导到客户，同时保证客户自身拿不到的市场信息不会经关联方“回流”，"
         "在客户只提供报表与风险事件的条件下，预测其未来12个月陷入经营困境的概率。", key)
    para(f"核心结果（2021–2023年时间外测试，每季度约{B['avg_clients_per_quarter']:,.0f}户、共{B['distressed_firms']}家困境企业）："
         f"关联图谱模型的排序能力优于传统报表评分卡与不用图谱的同特征模型（AUC {main_run['roc_auc']:.3f}对{run('gbdt_own_unlisted')['roc_auc']:.3f}、{run('mlp')['roc_auc']:.3f}，"
         f"KS {main_run['ks']:.3f}对{run('gbdt_own_unlisted')['ks']:.3f}、{run('mlp')['ks']:.3f}）；每季度复核风险最高的5%客户，可提前覆盖{pct(g('graph', .05))}的困境事件"
         f"（评分卡{pct(g('own_scorecard', .05))}）；覆盖一半困境事件所需的复核量由{pct(half['own_scorecard'], 0)}客户降至{pct(half['graph'], 0)}；"
         f"提前识别的困境企业{dm['graph']['firms']}家，比评分卡多{firms_gap}家（95%区间{firms_ci[0]:.0f}～{firms_ci[1]:.0f}家）。"
         f"图谱多识别的困境企业以中小企业为主，按债务金额计与评分卡相当（4.4节）。")

    # ---------------- 1 Data ----------------
    h('一、数据介绍')
    figure(framework(), '图1  方案总览：从多源数据到企业关联图谱，再到风险评分与业务处置。')
    table(['赛题要素', '本项目数据（WRDS 授权数据库）', '在图谱中的作用'],
          [['客户自身经营与财务', 'Compustat 年度时点财务（141万条快照）', '客户节点特征：偿债、盈利、流动性、规模'],
           ['上下游业务关联', 'Compustat 客户—供应商披露', '供货边；客户销售额占供应商收入的比例作为依赖度'],
           ['人员关联（董事、高管、实控人）', 'BoardEx 北美任职记录（74万条）', '同一人员同时在两家企业任职即连边'],
           ['股东与投资人关联', 'Thomson Reuters 13F 机构持股（≥5%持股66万条）', '同一机构对两家企业均持股≥5%即连边'],
           ['逾期、涉诉、失信等风险事件', 'Audit Analytics 诉讼、延迟披露、财务重述、破产；S&P 评级；CRSP 股价与退市', '节点风险事件时序；关联方市场信号；困境标签']],
          [105, 205, 185])
    para(f"样本覆盖2015–2023年北美全部上市企业，按季度建立图谱快照，共{len(snaps)}个季度、{A['nodes']:,}个企业-季度节点（每季度约{mean_edges('nodes'):,.0f}家）。"
         '每个季度末只使用当时已经公开的信息；标签为观察后12个月内是否陷入经营困境（破产申请或因业绩被摘牌）。'
         '模型在2015–2017年训练、2019年验证、2021–2023年测试，测试期完全不参与训练。')
    para('为贴合“经营类客户”这一赛题对象，我们采用非上市客户视角：客户本身只提供报表与风险事件，不使用其股价或外部评级；'
         '而它的关联方多为上市公司，其股价、评级与风险事件可以从公开渠道获得。')

    # ---------------- 2 Descriptive ----------------
    story.append(CondPageBreak(250)); h('二、描述性分析')
    years = A['by_year_rows_bankruptcy_distress']
    figure(vbars([(y, v[2] / v[0], ORANGE if y in ('2015', '2016', '2022', '2023') else BLUE) for y, v in years.items()],
                 '各年度企业未来12个月困境发生率', lambda v: f'{100 * v:.1f}%', height=135, max_value=0.025),
           '图2  困境发生率呈明显周期性：2015–2016年大宗商品下行与2022–2023年加息周期为两个高峰（橙色）。')
    dr = B['descriptive_distress_rate']
    figure(vbars([('无关联方', dr['无关联方'], '#b9c6d3'), ('有关联方\n无风险信号', dr['有关联方但无上述信号'], BLUE),
                  ('关联方\n延迟披露', dr['关联方延迟披露'], YELLOW), ('关联方股价\n一年跌超50%', dr['关联方股价一年跌超50%'], ORANGE),
                  ('关联方近一年\n已陷困境', dr['关联方近一年发生困境'], RED)],
                 '测试期客户未来12个月困境发生率：按关联方在观察时点的风险信号分组', lambda v: f'{100 * v:.2f}%', height=145, max_value=0.03),
           f"图3  风险会沿关联关系传导：关联方近一年已陷困境的客户，其困境发生率是“有关联方但无风险信号”客户的{dr['关联方近一年发生困境'] / dr['有关联方但无上述信号']:.1f}倍。")
    cov = U['relation_coverage_test']
    para(f"关联关系覆盖面广：测试期{pct(cov['people'])}的客户存在共享董事/高管的关联企业，{pct(cov['investor'])}有共同大股东，"
         f"{pct(cov['customer'])}有已披露的主要客户、{pct(cov['supplier'])}有已披露的供应商；平均每季度约{mean_edges('people'):,.0f}条人员关联、"
         f"{mean_edges('investor'):,.0f}条股东关联与{mean_edges('supply'):,.0f}条供应链关联。")
    fo = lambda k: F['runs'][k]['test']['ap']
    figure(vbars([('仅用报表', fo('gbdt_own_financial'), '#b9c6d3'), ('报表+自身股价\n（上市企业）', fo('gbdt_own_financial_market'), BLUE),
                  ('报表+风险事件\n（非上市客户）', run('mlp')['ap'], YELLOW), ('非上市客户\n+关联图谱', main_run['ap'], ORANGE)],
                 '不同信息条件下的困境识别能力（测试期 AP；随机猜测约 0.012）', lambda v: f'{v:.3f}', height=140, max_value=0.27),
           f"图4  自身股价是上市企业最强的风险信号；没有股价的经营类客户识别难度大得多。关联图谱为这类客户补回了一部分外部信息"
           f"（AP {run('mlp')['ap']:.3f}→{main_run['ap']:.3f}，AUC {run('mlp')['roc_auc']:.3f}→{main_run['roc_auc']:.3f}），但不能替代自身的市场信号——"
           "图谱模型的定位是在行内报表评分之上提供增量预警。")

    # ---------------- 3 Model ----------------
    story.append(CondPageBreak(330)); h('三、模型分析')
    h('3.1 企业关联图谱', 2)
    figure(graph_schema(), '图5  图谱结构：客户以“非上市视角”进入，关联方以完整信息进入；三类关系合并为一张关联网，两层传导覆盖“关联方的关联方”，且客户自身信息不会经关联方回流。')
    h('3.2 非回溯 GraphSAGE 图神经网络', 2)
    figure(architecture(), '图6  模型结构：客户与关联方使用不同编码器；两层 GraphSAGE 均值聚合把关联方及其关联方的风险传到客户；非回溯约束保证客户自身信息不回流。')
    para('三个关键设计：（1）双编码器——客户只用行内可得信息，关联方用全量公开信息，二者在同一表示空间中交互；'
         '（2）GraphSAGE 均值聚合——关联方的整体风险环境与客户自身表示分开加权，结构简洁，对新进件客户可直接推断，也便于扩展到行内大规模关联网络；'
         '（3）非回溯传导——客户作为别人的关联方时以完整信息入图，普通的两层传导会经“客户→关联方→客户”把客户自己的股价、评级带回给它；'
         '非回溯传导在第二层先从关联方的消息中剔除客户自身的信息，保证模型只使用银行对非上市客户真正可得的信息。')
    h('3.3 模型比较', 2)
    rows = [('gbdt_own_unlisted', '传统报表评分卡（GBDT）'), ('mlp', '同特征神经网络（不用图谱）'), ('two_stage_contagion', '两阶段关联特征模型'),
            ('sage_nb', '非回溯 GraphSAGE（本方案）'), ('sage_nb_rewired', '对照：关联关系随机打乱')]
    table(['模型（非上市客户视角，2021–2023测试）', '困境 AP', 'AUC', 'KS', '破产 AP'],
          [[n, f"{run(k)['ap']:.3f}", f"{run(k)['roc_auc']:.3f}", f"{run(k)['ks']:.3f}", f"{UB['runs'][k]['test']['ap']:.3f}"] for k, n in rows],
          [215, 70, 70, 70, 70], highlight=3,
          note=f"随机猜测的AP约等于事件发生率（困境{base:.3f}、破产{load('distress_bankruptcy_unlisted_gnn_study.json')['test_event_rate']:.3f}）。"
               f"本方案相对不用图谱的同特征模型：AUC {rng(vs_mlp, 'auc')}，AP {rng(vs_mlp)}；相对传统评分卡：AP {rng(vs_card)}，AUC {rng(vs_card, 'auc')}。"
               f"以随机打乱关联关系作对照，本方案的AP与之相当（本方案减对照{rng(vs_rand)}），AUC则显著更高（{rng(vs_rand, 'auc')}）：名单精度上的提升有相当部分来自同季度企业整体的市场环境，"
               f"真实关联关系的贡献主要体现在排序能力上。")
    roc = load('roc_curves.json')
    tpr_at = lambda k, x: roc['models'][k]['tpr'][roc['fpr'].index(x)]
    roc_color = lambda k: ORANGE if k == 'sage_nb' else ('#9a9890' if k.endswith('rewired') else BLUE)
    roc_titles = dict(rows)
    figure(roc_grid(roc, [(k, roc_titles[k], roc_color(k)) for k, _ in rows], 'sage_nb'),
           f"图7  各模型在测试期的ROC曲线（{roc['n']:,}个企业-季度，其中{roc['positives']:,}个在12个月内陷入困境）：横轴为误把健康客户列为高风险的比例，纵轴为困境客户被识别出的比例；"
           f"灰色虚线为随机猜测，浅橙色虚线为本方案。误报率5%时本方案识别出{pct(tpr_at('sage_nb', .05))}的困境客户，传统评分卡{pct(tpr_at('gbdt_own_unlisted', .05))}，"
           f"不用图谱的模型{pct(tpr_at('mlp', .05))}。")
    sr = S['runs']; sa = lambda k: sr[k]['test']
    figure(vbars([('仅供应链', sa('supply_only')['roc_auc'], BLUE), ('仅人员关系', sa('people_only')['roc_auc'], BLUE),
                  ('仅股东关系', sa('investor_only')['roc_auc'], BLUE), ('三类关系\n（本方案）', sa('main_all_relations_full_neighbour_view')['roc_auc'], ORANGE),
                  ('关联方也不用\n市场信号', sa('neighbour_view_unlisted_only')['roc_auc'], '#b9c6d3')],
                 '消融分析：不同关系与关联方信息对排序能力的贡献（测试期 AUC）', lambda v: f'{v:.3f}', height=135, max_value=1.0),
           f"图8  三类关系合用时AUC最高，单用供应链明显偏弱（{sa('supply_only')['roc_auc']:.3f}）；若关联方也只提供报表信息，AUC降至{sa('neighbour_view_unlisted_only')['roc_auc']:.3f}——"
           f"上市关联方的市场信号是风险传导的主要载体。按AP比较，各消融版本与本方案的差异均不显著（单用股东关系为{sa('investor_only')['ap']:.3f}，本方案{sa('main_all_relations_full_neighbour_view')['ap']:.3f}）。")

    # ---------------- 4 Validation: what the model measurably did in the test period ----------------
    h('四、应用验证')
    h('4.1 预警效果：用更少的复核覆盖更多风险', 2)
    para(f"本节用两个业务指标衡量模型：“名单命中率”——名单里有多少客户真的出事；“抓获率”——出事的客户有多少提前进了名单。"
         f"如何理解AP={main_run['ap']:.3f}：其下限等于事件发生率（约{base:.3f}），困境是年发生率约1%的稀有事件，AP {main_run['ap']:.3f}约为随机水平的{main_run['ap'] / base:.0f}倍。"
         f"换成业务语言：每季度风险最高的1%客户中，约每{1 / prec('graph', .01):.0f}户就有1户在一年内陷入困境，而随机抽查要约{1 / base:.0f}户才遇到1户。")
    figure(gains_chart(shares, [('关联图谱模型', gains['graph'], ORANGE, None), ('不用图谱的模型', gains['no_graph'], BLUE, None),
                                ('传统报表评分卡', gains['own_scorecard'], AQUA, [3, 2]), ('随机抽查', gains['random'], MUTED, [1, 2])]),
           f"图9  预警名单的“抓获率”曲线：每季度复核前5%客户（虚线），本方案抓到{pct(g('graph', .05))}的困境事件，不用图谱{pct(g('no_graph', .05))}，传统评分卡{pct(g('own_scorecard', .05))}，随机抽查5%。")
    table(['业务指标（2021–2023测试期，按季度滚动）', '关联图谱模型', '不用图谱', '传统评分卡'],
          [['复核前5%客户：抓到的困境事件', pct(g('graph', .05)), pct(g('no_graph', .05)), pct(g('own_scorecard', .05))],
           ['复核前5%客户：名单命中率（倍于随机）', f"{pct(prec('graph', .05))}（{prec('graph', .05) / base:.1f}倍）", f"{pct(prec('no_graph', .05))}（{prec('no_graph', .05) / base:.1f}倍）", f"{pct(prec('own_scorecard', .05))}（{prec('own_scorecard', .05) / base:.1f}倍）"],
           ['复核前1%客户：名单命中率', pct(prec('graph', .01)), pct(prec('no_graph', .01)), pct(prec('own_scorecard', .01))],
           ['抓到一半困境事件所需复核比例', pct(half['graph'], 0), pct(half['no_graph'], 0), pct(half['own_scorecard'], 0)],
           ['困境企业在事发前被列入前5%名单的比例', pct(B['lead_time']['graph']['firms_flagged_share']), pct(B['lead_time']['no_graph']['firms_flagged_share']), pct(B['lead_time']['own_scorecard']['firms_flagged_share'])],
           ['其中提前6个月以上预警的比例', pct(B['lead_time']['graph']['flagged_6m_or_earlier_share']), pct(B['lead_time']['no_graph']['flagged_6m_or_earlier_share']), pct(B['lead_time']['own_scorecard']['flagged_6m_or_earlier_share'])]],
          [215, 95, 90, 95], highlight=0)
    para(f"预警提前期：被预警的困境企业进入名单的时间中位数约在事发前{B['lead_time']['graph']['median_lead_months_when_flagged']:.1f}个月，留出压降、追加担保与重组的时间窗口。"
         f"图谱带来的是增量而非替代：{B['caught_only_with_graph']}家困境企业只有关联图谱模型能提前识别，只有不用图谱的模型能识别的为{B['caught_only_without_graph']}家。")

    h('4.2 评分等级：可直接用于授信分层', 2)
    card = S['scorecard']['grades']; rows_all = sum(x['test_rows'] for x in card); ev_all = sum(x['test_events'] for x in card)
    figure(vbars([(f"{gr['grade']}级\n{gr['test_rows'] / rows_all * 100:.1f}%客户", gr['test_event_rate'], RED if gr['grade'] in 'ED' else (YELLOW if gr['grade'] == 'C' else AQUA)) for gr in card],
                 '评分等级与测试期困境发生率（600分基准，分越高越安全；等级切点由训练期确定）', lambda v: f'{100 * v:.1f}%', height=140, max_value=0.15),
           f"图10  等级区分度清晰且严格单调：E级困境率{pct(card[0]['test_event_rate'])}，A级仅{pct(card[4]['test_event_rate'], 2)}，相差约{card[0]['test_event_rate'] / card[4]['test_event_rate']:.0f}倍；"
           f"D、E两级合计{pct((card[0]['test_rows'] + card[1]['test_rows']) / rows_all)}的客户覆盖了{pct((card[0]['test_events'] + card[1]['test_events']) / ev_all)}的困境事件。")

    h('4.3 可解释性：说清风险从哪里来', 2)
    if B.get('case'):
        figure(ego_case(B['case']), f"图11  传导路径示例（真实测试样本，已匿名；选自“图谱模型列入前5%、不用图谱时排在同季前25%以外”的{B['case']['candidates']}家困境企业）："
                                    "客户自身报表风险不突出，但多个共享董事/高管的关联企业股价一年内大幅下跌，模型提前将其列入复核名单。")
    at = S['relation_attribution_top5pct']; ds = at['dominant_relation_share']
    para(f"对每季度前5%的高风险客户，逐类移除关联关系观察风险分变化，可定位风险来源——{pct(ds['people'])}的高风险客户主要风险来自人员关联（共享董事/高管，常对应同一集团或实控人），"
         f"{pct(ds['investor'])}来自共同大股东，{pct(ds['supplier'] + ds['customer'])}来自上下游；对{pct(at['share_where_relations_explain_over_half'])}的高风险客户，关联关系贡献了一半以上的风险分。")

    h('4.4 业务价值：以实测债务衡量', 2)
    para("把每季度前5%名单中“事后确实陷入困境”的企业找出来，按企业去重，取其首次进入名单那个季度、观察时点已披露的有息债务"
         "（短期+长期借款，Compustat 时点财务）。这是企业对全部债权人的总债务，不是单家银行的敞口；样本为北美上市企业。")
    table(['实测业务价值（测试期合计）', '关联图谱模型', '不用图谱', '传统评分卡'],
          [['提前识别的困境企业（家）', f"{dm['graph']['firms']}", f"{dm['no_graph']['firms']}", f"{dm['own_scorecard']['firms']}"],
           ['其合计债务', yi(dm['graph']['debt_total_nominal_m']), yi(dm['no_graph']['debt_total_nominal_m']), yi(dm['own_scorecard']['debt_total_nominal_m'])],
           ['按金额计的抓获率', pct(dm['graph']['capture_by_amount']), pct(dm['no_graph']['capture_by_amount']), pct(dm['own_scorecard']['capture_by_amount'])]],
          [175, 105, 105, 110], highlight=0,
          note=f"图谱模型比评分卡多识别{firms_gap}家困境企业（企业重抽样95%区间{firms_ci[0]:.0f}～{firms_ci[1]:.0f}家）；按金额计与评分卡相当"
               f"（差{yi(D['graph_minus_model_m']['own_scorecard'])}，区间{ci(conc['graph_minus_model_ci95_m']['own_scorecard'])}），低于不用图谱的模型"
               f"（差{yi(D['graph_minus_model_m']['no_graph'])}，区间{ci(conc['graph_minus_model_ci95_m']['no_graph'])}）。图谱多识别的主要是中小企业，"
               f"困境企业债务高度集中（中位数约{conc['median_firm_debt_m']:.0f}百万美元，最大的10家占{pct(conc['largest10_share_of_all_distressed_debt'], 0)}），"
               "少数大额困境企业靠自身报表已能识别。金额为企业对所有债权人的总有息债务（测试期合计，非年值）。")
    rnd = dm['random']
    figure(vbars([(f"随机抽查5%\n（期望约{rnd['firms']:.0f}家）", rnd['firms'], '#b9c6d3'),
                  (f"传统评分卡", dm['own_scorecard']['firms'], AQUA), (f"不用图谱的模型", dm['no_graph']['firms'], BLUE),
                  (f"关联图谱模型", dm['graph']['firms'], ORANGE)],
                 '2021–2023年测试期：复核前5%名单提前识别的困境企业家数（按企业去重，实测）', lambda v: f'{v:,.0f}', height=140, max_value=220),
           f"图12  口径：每季度按风险分复核前5%客户；困境企业按企业去重；随机抽查为期望值。全部困境企业{D['all_distressed']['firms']}家，"
           f"合计债务约{yi(D['all_distressed']['debt_total_nominal_m'])}。")

    # ---------------- 5 Product ----------------
    h('五、产品思路：关联风险雷达')
    h('5.1 痛点与定位', 2)
    para('产品定位：“关联风险雷达”——面向银行经营类客户的关联风险评分与预警系统。赛题指出经营类客户的风险易受上下游、股东、实控人等关联主体传导，'
         '而银行对客户本身的数据较充分、对其关联方的动态却难以及时掌握。系统把关联方的风险变化自动传递到客户层面，作为行内报表评分之上的增量预警，嵌入贷前、贷中、贷后三个环节。')
    h('5.2 嵌入授信全流程', 2)
    table(['业务环节', '系统功能', '对应模型输出'],
          [['贷前准入与定价', '新客户进件时自动拉取其关联方网络，出具“自身+关联”综合评分与等级', 'A–E 等级；关联风险贡献占比'],
           ['贷中预警', '每季度（可提升为月度）重算全量客户；关联方出现股价暴跌、延迟披露、涉诉、破产时触发重评', '前5%高风险名单；等级下迁提醒'],
           ['贷后管理', '对 D/E 级客户推送传导路径：风险来自哪类关系、哪个关联方、何种事件', '关系归因；高风险关联方列表'],
           ['组合管理', '识别同一高风险关联方牵连的客户群，防范“一点爆雷、多户受损”', '共同关联方聚类；集中度预警']],
          [80, 290, 125])
    h('5.3 投入产出估算', 2)
    table(['投入产出（测试期实测）', '数值', '说明'],
          [['覆盖一半困境事件所需复核比例', f"{pct(half['graph'], 0)} 对 {pct(half['own_scorecard'], 0)}", f"复核工作量较评分卡减少约{pct(1 - half['graph'] / half['own_scorecard'], 0)}"],
           ['提前识别的困境企业', f"{dm['graph']['firms']} 对 {dm['own_scorecard']['firms']} 家", f"多{firms_gap}家（区间{firms_ci[0]:.0f}～{firms_ci[1]:.0f}家），以中小企业为主"],
           ['按债务金额计', '与评分卡相当', '大额困境企业靠自身报表已能识别，不宜以金额口径换算损失规避'],
           ['建议用法', '与行内评分卡并行', '评分卡管大额客户，关联雷达补足中小经营类客户的关联风险']],
          [150, 120, 225], highlight=1)
    h('5.4 落地路径', 2)
    para('推广建议：以行内工商股权、担保圈、对公结算识别的上下游与征信事件替换公开数据，按同一框架重建图谱，并以2024年以后数据做前瞻验证后分阶段上线。'
         '上线时保留非回溯约束；集团股权、担保与贷款关系是公开数据缺少、而最可能放大图谱价值的数据。')

    # ---------------- 6 Conclusion ----------------
    h('六、总结结论')
    para(f"（1）构建了覆盖供应链、董事高管与大股东三类关系、携带涉诉/延迟披露/重述/破产等事件时序的企业关联图谱（{A['nodes']:,}个企业-季度节点），并提出面向非上市经营类客户的非回溯 GraphSAGE 图神经网络。")
    para(f"（2）关联风险真实存在：关联方已陷困境的客户困境率是无风险信号客户的{dr['关联方近一年发生困境'] / dr['有关联方但无上述信号']:.1f}倍；人员（集团/实控人）关联是高风险客户最主要的风险来源；"
         f"关联方的市场信号是风险传导的主要载体（关联方只用报表信息时AUC由{sa('main_all_relations_full_neighbour_view')['roc_auc']:.3f}降至{sa('neighbour_view_unlisted_only')['roc_auc']:.3f}）。")
    para(f"（3）在严格的时间外测试中，模型AUC {main_run['roc_auc']:.3f}、KS {main_run['ks']:.3f}，排序能力显著优于不用图谱的同特征模型；每季度复核5%客户可提前覆盖{pct(g('graph', .05))}的困境事件，"
         f"复核工作量较传统评分卡减少约{pct(1 - half['graph'] / half['own_scorecard'], 0)}，多识别困境企业{firms_gap}家。")
    para("（4）关联图谱的增益稳健但温和，且有一部分来自市场整体环境；多识别的困境企业以中小经营类客户为主。关联图谱更适合作为行内评分卡之上的增量预警，"
         "其价值有望随行内股权、担保、贷款等关系数据的接入而放大。")
    story.append(PageMarker(pages, 'body_end'))

    def footer(canvas, doc):
        # Paper-style running footer: hairline rule, code-availability note with a clickable link, page number.
        left, right, y = 50, A4[0] - 50, 22
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor(GRID)); canvas.setLineWidth(0.5); canvas.line(left, y + 10, right, y + 10)
        canvas.setFont('ReportCJK', 7); canvas.setFillColor(colors.HexColor(MUTED))
        label = '代码与复现材料：'; canvas.drawString(left, y, label)
        x = left + pdfmetrics.stringWidth(label, 'ReportCJK', 7)
        canvas.setFillColor(colors.HexColor(BLUE)); canvas.drawString(x, y, REPO_URL)
        canvas.linkURL(REPO_URL, (x, y - 2, x + pdfmetrics.stringWidth(REPO_URL, 'ReportCJK', 7), y + 7), relative=0)
        canvas.setFillColor(colors.HexColor(MUTED)); canvas.setFont('ReportCJK', 7.5); canvas.drawRightString(right, y, f'{doc.page}')
        canvas.restoreState()
    SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=50, rightMargin=50, topMargin=42, bottomMargin=40,
                      title='基于关联图谱的经营类客户风险预警模型').build(story, onFirstPage=footer, onLaterPages=footer)
    MD.write_text(''.join(md) + f'---\n\n代码与复现材料：{REPO_URL}\n')
    print(json.dumps({'pdf': str(PDF), 'body_pages': pages.get('body_end'), 'within_limit': pages.get('body_end', 99) <= BODY_PAGE_LIMIT}, ensure_ascii=False))
    if pages.get('body_end', 99) > BODY_PAGE_LIMIT:
        raise SystemExit('Body exceeds page limit')


if __name__ == '__main__':
    main()
