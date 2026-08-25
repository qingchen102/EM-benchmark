"""读取 xlsx（纯标准库 zipfile + XML），输出各 sheet 内容摘要。"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx(path):
    z = zipfile.ZipFile(path)
    # 共享字符串表
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall(f"{NS}si"):
            txt = "".join(t.text or "" for t in si.iter(f"{NS}t"))
            shared.append(txt)
    # workbook: sheet 名与 rId
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    sheets = []
    for sh in wb.iter(f"{NS}sheet"):
        sheets.append((sh.get("name"), sh.get(f"{{http://schemas.openxmlformats.org/officeDocument/2006/relationships}}id")))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rel_map = {r.get("Id"): r.get("Target") for r in rels}
    result = {}
    for name, rid in sheets:
        target = rel_map.get(rid, "")
        if not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")
        rows = []
        root = ET.fromstring(z.read(target))
        sheet_data = root.find(f"{NS}sheetData")
        if sheet_data is None:
            continue
        for row in sheet_data.findall(f"{NS}row"):
            cells = {}
            for c in row.findall(f"{NS}c"):
                ref = c.get("r", "")
                col = re.match(r"[A-Z]+", ref).group(0) if ref else ""
                t = c.get("t")
                v = c.find(f"{NS}v")
                if t == "s" and v is not None:
                    val = shared[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(f"{NS}t"))
                elif v is not None:
                    val = v.text
                else:
                    val = ""
                cells[col] = val
            if cells:
                rows.append(cells)
        result[name] = rows
    return result


if __name__ == "__main__":
    data = read_xlsx(sys.argv[1])
    for sheet, rows in data.items():
        print(f"=== [{sheet}] {len(rows)} 行 ===")
        # 列顺序按首行出现顺序
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        # 打印最多 15 行
        for r in rows[:15]:
            print(" | ".join(f"{k}:{r.get(k,'')[:40]}" for k in cols))
        print()
