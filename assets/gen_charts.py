#!/usr/bin/env python3
"""Performance-history charts for the recipe README: static SVG, one light and one dark file per chart.

usage: gen_charts.py DATA.json OUTDIR
Colors follow the dataviz reference palette: releases are an ordered set, so they use one validated ordinal blue ramp
(darker = newer on light, lighter = newer on dark); the time-breakdown chart uses categorical slots 1-2 plus a neutral.
"""
import json, sys, os
from html import escape

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
THEME = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  ramp=['#86b6ef', '#5598e7', '#256abf', '#104281'], verify='#2a78d6', draft='#eb6834', idle='#c3c2b7',
                  good='#006300', border='rgba(11,11,11,0.10)'),
    'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 ramp=['#184f95', '#2a78d6', '#5598e7', '#9ec5f4'], verify='#3987e5', draft='#d95926', idle='#52514e',
                 good='#0ca30c', border='rgba(255,255,255,0.10)'),
}


def fmt(v, digits=0):
    return f'{v + 1e-9:,.{digits}f}'  # round half up (69.05 -> 69.1, not binary-float 69.0)


class Svg:
    def __init__(self, w, h, title, desc, t):
        self.w, self.h, self.t = w, h, t
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
                      f'aria-labelledby="t d" font-family=\'{FONT}\'>',
                      f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>',
                      f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="12" fill="{t["surface"]}" stroke="{t["border"]}"/>']

    def text(self, x, y, s, size=12.0, color=None, anchor='start', weight=400, extra=''):
        c = color or self.t['ink2']
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{c}" text-anchor="{anchor}" '
                          f'font-weight="{weight}" {extra}>{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, color, width=1):
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>')

    def column(self, x, y_top, w, base, color, r=4):
        h = base - y_top
        r = min(r, h, w / 2)
        self.parts.append(f'<path d="M{x:.1f},{base:.1f} V{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} '
                          f'H{x + w - r:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} V{base:.1f} Z" fill="{color}"/>')

    def hbar(self, x0, y, x1, h, color, round_end):
        if x1 - x0 <= 0:
            return
        r = min(4, (x1 - x0) / 2, h / 2) if round_end else 0
        self.parts.append(f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} V{y + h - r:.1f} '
                          f'Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{color}"/>')

    def raw(self, s):
        self.parts.append(s)

    def save(self, path):
        open(path, 'w').write('\n'.join(self.parts + ['</svg>']) + '\n')


def legend(svg, x, y, items, t):
    """items: [(label, color)] -> swatches with text in secondary ink; returns end x."""
    for label, color in items:
        svg.raw(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" fill="{color}"/>')
        svg.text(x + 15, y, label, 12, t['ink2'])
        x += 15 + 7.2 * len(label) + 18
    return x


def small_multiples(data, key, title, subtitle, unit_digits, ymax, yticks, path, mode, note):
    t = THEME[mode]
    rel = data['releases']
    panels = data[key]
    W, H = 900, 360
    left, right, top, bottom = 58, 18, 92, 64
    pw = (W - left - right) / len(panels)
    base = H - bottom
    ph = base - top
    svg = Svg(W, H, title, subtitle + '. ' + note, t)
    svg.text(24, 32, title, 17, t['ink'], weight=600)
    svg.text(24, 52, subtitle, 12.5, t['ink2'])
    legend(svg, 24, 76, [(r['label'], t['ramp'][i]) for i, r in enumerate(rel)], t)
    for v in yticks:
        y = base - ph * v / ymax
        svg.line(left, y, W - right, y, t['grid'] if v else t['axis'])
        svg.text(left - 8, y + 4, fmt(v), 11, t['muted'], 'end', extra='font-variant-numeric="tabular-nums"')
    cw, gap = 22, 14
    for pi, p in enumerate(panels):
        px = left + pi * pw
        group_w = len(rel) * cw + (len(rel) - 1) * gap
        x0 = px + (pw - group_w) / 2
        for i, r in enumerate(rel):
            v = p['values'].get(r['id'])
            if v is None:
                continue
            x = x0 + i * (cw + gap)
            yt = base - ph * v / ymax
            svg.column(x, yt, cw, base, t['ramp'][i])
            svg.text(x + cw / 2, yt - 6, fmt(v, unit_digits), 10.5, t['ink2'], 'middle')
        svg.text(px + pw / 2, base + 20, p['label'], 12, t['ink'], 'middle', weight=600)
        first, last = p['values'].get(rel[0]['id']), p['values'].get(rel[-1]['id'])
        if first and last:
            svg.text(px + pw / 2, base + 37, f'{rel[0]["short"]} → {rel[-1]["short"]}: {last / first:.2f}×', 11.5, t['ink2'], 'middle')
    svg.text(W - right, H - 10, note, 10.5, t['muted'], 'end')
    svg.save(path)


def timeline(data, path, mode):
    t = THEME[mode]
    pts = data['timeline']
    W, H = 900, 340
    left, right, top, bottom = 58, 40, 70, 78
    base = H - bottom
    ph = base - top
    ymax = 160
    title = 'Decode speed across recipe releases'
    sub = 'tokens/s on the 7.3K-token /completion benchmark (512 tokens, temperature 0)'
    svg = Svg(W, H, title, sub + '. ' + data['timeline_note'], t)
    svg.text(24, 32, title, 17, t['ink'], weight=600)
    svg.text(24, 52, sub, 12.5, t['ink2'])
    for v in (0, 40, 80, 120, 160):
        y = base - ph * v / ymax
        svg.line(left, y, W - right, y, t['grid'] if v else t['axis'])
        svg.text(left - 8, y + 4, fmt(v), 11, t['muted'], 'end', extra='font-variant-numeric="tabular-nums"')
    n = len(pts)
    xs = [left + 40 + i * (W - left - right - 80) / (n - 1) for i in range(n)]
    ys = [base - ph * p['value'] / ymax for p in pts]
    ref = data.get('timeline_ref')
    if ref:  # stock llama.cpp on the September model: only meaningful across the points that use that model
        i0 = ref['from_index']
        y = base - ph * ref['value'] / ymax
        svg.line(xs[i0] - 34, y, xs[-1] + 26, y, t['muted'])
        svg.text(xs[i0] - 34, y + 15, f'{ref["label"]}: {fmt(ref["value"], 1)}', 10.5, t['muted'])
    color = t['ramp'][3]
    d = ' '.join(f'{"M" if i == 0 else "L"}{x:.1f},{y:.1f}' for i, (x, y) in enumerate(zip(xs, ys)))
    svg.raw(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for i, (x, y, p) in enumerate(zip(xs, ys, pts)):
        svg.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')
        svg.text(x, y - 12, fmt(p['value'], 1), 11.5, t['ink'], 'middle', weight=600 if i == n - 1 else 400)
        svg.text(x, base + 20, p['label'], 12, t['ink'], 'middle', weight=600)
        svg.text(x, base + 36, p['sub'], 10.5, t['ink2'], 'middle')
        if p.get('sub2'):
            svg.text(x, base + 50, p['sub2'], 10.5, t['muted'], 'middle')
    svg.text(W - 18, H - 8, data['timeline_note'], 10.5, t['muted'], 'end')
    svg.save(path)


def breakdown(data, path, mode):
    t = THEME[mode]
    rows = data['breakdown']
    W = 900
    left, right, top = 150, 150, 92
    bh, rg = 22, 18
    H = top + len(rows) * (bh + rg) + 46
    xmax = data['breakdown_xmax']
    pw = W - left - right
    title = 'Where each generated token’s time goes'
    sub = 'milliseconds of decode wall time per generated token on real agent sessions (lower is better)'
    svg = Svg(W, H, title, sub + '. ' + data['breakdown_note'], t)
    svg.text(24, 32, title, 17, t['ink'], weight=600)
    svg.text(24, 52, sub, 12.5, t['ink2'])
    legend(svg, 24, 76, [('verify pass (target model)', t['verify']), ('MTP drafting', t['draft']),
                         ('GPU idle (host work, syncs)', t['idle'])], t)
    for v in range(0, int(xmax) + 1, 2):
        x = left + pw * v / xmax
        svg.line(x, top - 6, x, top + len(rows) * (bh + rg) - rg + 6, t['grid'] if v else t['axis'])
        svg.text(x, top + len(rows) * (bh + rg) - rg + 22, f'{v} ms', 11, t['muted'], 'middle')
    for i, r in enumerate(rows):
        y = top + i * (bh + rg)
        svg.text(left - 12, y + bh / 2 + 4, r['label'], 12.5, t['ink'], 'end', weight=600)
        x = left
        segs = [(r['verify'], t['verify']), (r['draft'], t['draft']), (r['idle'], t['idle'])]
        for j, (v, c) in enumerate(segs):
            x1 = x + pw * v / xmax
            last = j == len(segs) - 1
            svg.hbar(x, y, (x1 - 2) if not last else x1, bh, c, last)
            x = x1
        total = r['verify'] + r['draft'] + r['idle']
        svg.text(x + 10, y + bh / 2 + 4, f'{total:.2f} ms  ·  {1000 / total:.0f} tok/s', 12, t['ink2'])
    svg.text(W - 18, H - 10, data['breakdown_note'], 10.5, t['muted'], 'end')
    svg.save(path)


def main():
    data = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for mode in ('light', 'dark'):
        small_multiples(data, 'decode', 'Decode speed by release', 'tokens/s, higher is better · same model, machine and session, 96K context',
                        0, 160, (0, 40, 80, 120, 160), f'{out}/decode-{mode}.svg', mode, data['session_note'])
        small_multiples(data, 'prefill', 'Prompt processing by release', 'tokens/s, higher is better · same model, machine and session, 96K context',
                        0, 2400, (0, 600, 1200, 1800, 2400), f'{out}/prefill-{mode}.svg', mode, data['session_note'])
        timeline(data, f'{out}/timeline-{mode}.svg', mode)
        if 'breakdown' in data:
            breakdown(data, f'{out}/breakdown-{mode}.svg', mode)


if __name__ == '__main__':
    main()
