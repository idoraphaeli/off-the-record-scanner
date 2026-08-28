# -*- coding: utf-8 -*-
"""
Wrap label_tool/index.json as a plain script that assigns a global.

The tool is opened straight off disk, and a browser refuses fetch() on a
file:// URL, so the page cannot read the JSON as data. A <script src> is not
subject to that rule, so the same content is served as an assignment instead.

Usage:  python make_index_js.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "label_tool")

with open(os.path.join(TOOL, "index.json"), encoding="utf-8") as fh:
    data = json.load(fh)

with open(os.path.join(TOOL, "index.js"), "w", encoding="utf-8") as fh:
    fh.write("window.LABEL_DATA = ")
    json.dump(data, fh, ensure_ascii=False)
    fh.write(";\n")

print(f"{data['total']} detections to label -> {os.path.join(TOOL, 'index.js')}")
