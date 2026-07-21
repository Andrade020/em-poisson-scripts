# -*- coding: utf-8 -*-
"""Baixa e identifica os arquivos DATATRAN (por ocorrencia) via Google Drive."""
import os
import re
import zipfile
import urllib.request

os.makedirs("dt_probe", exist_ok=True)
ids = [l.strip() for l in open("drive_ids.txt") if l.strip()]
print(len(ids), "ids")
found = {}
for gid in ids:
    dest = f"dt_probe/{gid}"
    if not os.path.exists(dest):
        url = f"https://drive.google.com/uc?export=download&id={gid}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=240) as r, open(dest, "wb") as f:
                f.write(r.read())
        except Exception as e:
            print(gid, "ERRO", str(e)[:60])
            continue
    try:
        names = zipfile.ZipFile(dest).namelist()
        print(gid, names[:1], round(os.path.getsize(dest) / 1e6, 1), "MB")
        m = re.match(r"datatran(\d{4})\.csv", names[0]) if names else None
        if m:
            found[m.group(1)] = gid
            os.replace(dest, f"datatran_{m.group(1)}_ok.zip")
    except Exception:
        os.remove(dest)
print("ANOS ENCONTRADOS:", sorted(found))
