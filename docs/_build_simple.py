# -*- coding: utf-8 -*-
"""Build a print-ready A4 HTML + PDF for the simplified planning note."""
import pathlib, subprocess, os, re, sys
import mistune

ROOT = pathlib.Path(r"D:\srtp\EM-benchmark\docs")
SRC = ROOT / "低空公司采集规划_简版_2026-10-23.md"
HTML = ROOT / "_simple.html"
PDF = ROOT / "低空公司采集规划_简版_2026-10-23.pdf"

text = SRC.read_text(encoding="utf-8")
body = mistune.create_markdown(plugins=["table", "strikethrough"])(text)

def decorate(html: str) -> str:
    def fix(m):
        t = m.group(0)
        th = re.search(r"<th[^>]*>(.*?)</th>", t, re.S)
        head = re.sub(r"<[^>]+>", "", th.group(1)).strip() if th else ""
        cls = {"项": "cover", "项目": "cmp", "方案": "site", "模式": "mode",
               "优先级": "prio", "信号": "sig", "#": "chk"}.get(head)
        return t.replace("<table>", f'<table class="{cls}">', 1) if cls else t
    html = re.sub(r"<table>.*?</table>", fix, html, flags=re.S)
    return html.replace("<h2>4. 待确认清单</h2>",
                        '<h2 class="newpage">4. 待确认清单</h2>')

body = decorate(body)

CSS = r"""
@page { size: A4 portrait; margin: 14mm 12mm 12mm 12mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Microsoft YaHei", "微软雅黑", "PingFang SC", sans-serif;
  font-size: 9.4pt; line-height: 1.55; color: #111; margin: 0;
}
h1 { font-size: 18pt; line-height: 1.3; margin: 0 0 3mm 0; letter-spacing: .2pt; }
h2 { font-size: 12.5pt; margin: 5mm 0 1.8mm 0; padding-bottom: 1mm;
     border-bottom: .7pt solid #9aa3ad; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 3.5mm 0 1.3mm 0; page-break-after: avoid; }
p { margin: 1.1mm 0; }
ul, ol { margin: 1mm 0 1.5mm 0; padding-left: 5.5mm; }
li { margin: .7mm 0; }
hr { border: 0; border-top: .6pt solid #c9ced4; margin: 4mm 0; }
table { width: 100%; border-collapse: collapse; margin: 2mm 0 3.2mm 0;
        font-size: 7.2pt; line-height: 1.3; table-layout: fixed; }
th, td { border: .5pt solid #aeb5bd; padding: .9mm 1.1mm; vertical-align: top;
         overflow-wrap: anywhere; word-break: normal; }
th { background: #e9edf2; font-weight: 600; text-align: left; }
tr:nth-child(even) td { background: #fbfcfd; }
tr { page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f1f3f5;
       padding: 0 .5mm; }
strong { font-weight: 600; }
table.cover th:nth-child(1) { width: 16%; }
table.cover th:nth-child(2) { width: 84%; }
table.cmp th:nth-child(1) { width: 20%; }
table.cmp th:nth-child(2), table.cmp th:nth-child(3) { width: 40%; }
table.cmp { page-break-inside: avoid; }
table.sta th:nth-child(1) { width: 9%; }
table.sta th:nth-child(2) { width: 22%; }
table.sta th:nth-child(3) { width: 29%; }
table.sta th:nth-child(4) { width: 25%; }
table.sta th:nth-child(5) { width: 15%; }
table.prio th:nth-child(1) { width: 6%; }
table.prio th:nth-child(2) { width: 10%; }
table.prio th:nth-child(3) { width: 16%; }
table.prio th:nth-child(4) { width: 27%; }
table.prio th:nth-child(5) { width: 19%; }
table.prio th:nth-child(6) { width: 14%; }
table.prio th:nth-child(7) { width: 8%; }
table.site th:nth-child(1) { width: 10%; }
table.site th:nth-child(2) { width: 20%; }
table.site th:nth-child(3) { width: 14%; }
table.site th:nth-child(4) { width: 28%; }
table.site th:nth-child(5) { width: 28%; }
table.mode th:nth-child(1) { width: 14%; }
table.mode th:nth-child(2) { width: 20%; }
table.mode th:nth-child(3) { width: 20%; }
table.mode th:nth-child(4) { width: 28%; }
table.mode th:nth-child(5) { width: 18%; }
table.sig th:nth-child(1) { width: 15%; }
table.sig th:nth-child(2) { width: 24%; }
table.sig th:nth-child(3) { width: 22%; }
table.sig th:nth-child(4) { width: 22%; }
table.sig th:nth-child(5) { width: 17%; }
table.chk th:nth-child(1) { width: 4%; }
table.chk th:nth-child(2) { width: 12%; }
table.chk th:nth-child(3) { width: 52%; }
table.chk th:nth-child(4) { width: 32%; }
h2.newpage { margin-top: 5mm; }
table.chk { page-break-inside: avoid; }
"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>低空电磁数据采集规划（简版）</title>
<style>{CSS}</style></head>
<body>
{body}
</body></html>
"""

HTML.write_text(html, encoding="utf-8")

chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(chrome):
    chrome = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
       f"--print-to-pdf={PDF}", HTML.as_uri(), "--virtual-time-budget=3000"]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
print("exit", r.returncode)
print((r.stderr or "")[-600:])
if PDF.exists():
    print("PDF", PDF, PDF.stat().st_size, "bytes")
else:
    sys.exit("PDF not created")
