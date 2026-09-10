// minimal RFC-4180-ish CSV parser (quotes, embedded commas/newlines). Header row.
export interface ParsedCsv {
  header: string[];
  rows: string[][];
}

export function parseCSV(text: string, opts: { maxRows?: number } = {}): ParsedCsv {
  const maxRows = opts.maxRows ?? Infinity;
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let i = 0;
  let q = false;
  const n = text.length;
  const push = () => {
    row.push(field);
    field = "";
  };
  const eol = () => {
    push();
    rows.push(row);
    row = [];
  };
  while (i < n) {
    const c = text[i];
    if (q) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        q = false; i++; continue;
      }
      field += c; i++; continue;
    }
    if (c === '"') { q = true; i++; continue; }
    if (c === ",") { push(); i++; continue; }
    if (c === "\r") { i++; continue; }
    if (c === "\n") {
      eol(); i++;
      if (rows.length > maxRows + 1) break;
      continue;
    }
    field += c; i++;
  }
  if (field.length || row.length) eol();
  const header = (rows.shift() ?? []).map((h) => h.trim());
  return {
    header,
    rows: rows.filter((r) => r.length && !(r.length === 1 && r[0] === "")),
  };
}

export interface CsvSummary {
  rowCount: number;
  colCount: number;
  timeAxis: string;
  spanSeconds: number | null;
  estWindows: number | null;
  families: [string, number][];
  protocols: [string, number][];
}

export function summarise(header: string[], rows: string[][]): CsvSummary {
  const idx = (name: string) => header.indexOf(name);
  const tCol =
    idx("flow_start_epoch") >= 0
      ? idx("flow_start_epoch")
      : idx("Timestamp") >= 0
        ? idx("Timestamp")
        : -1;
  let tMin = Infinity;
  let tMax = -Infinity;
  let tKind = "none";
  if (tCol >= 0) {
    tKind = header[tCol];
    for (const r of rows) {
      const raw = r[tCol];
      const v =
        header[tCol] === "flow_start_epoch"
          ? parseFloat(raw)
          : Date.parse(raw) / 1000;
      if (Number.isFinite(v)) {
        tMin = Math.min(tMin, v);
        tMax = Math.max(tMax, v);
      }
    }
  }
  const span =
    Number.isFinite(tMin) && Number.isFinite(tMax) ? Math.max(0, tMax - tMin) : null;

  const famCol = idx("Label") >= 0 ? idx("Label") : idx("attack_family");
  const fam: Record<string, number> = {};
  if (famCol >= 0)
    for (const r of rows) {
      const k = (r[famCol] || "BENIGN").trim() || "BENIGN";
      fam[k] = (fam[k] || 0) + 1;
    }

  const proto: Record<string, number> = {};
  const pc = idx("Protocol");
  if (pc >= 0)
    for (const r of rows) {
      const k =
        ({ "6": "TCP", "17": "UDP" } as Record<string, string>)[String(r[pc]).trim()] ||
        r[pc];
      proto[k] = (proto[k] || 0) + 1;
    }

  return {
    rowCount: rows.length,
    colCount: header.length,
    timeAxis: tKind,
    spanSeconds: span,
    estWindows: span != null ? Math.floor(span / 10) : null,
    families: Object.entries(fam).sort((a, b) => b[1] - a[1]),
    protocols: Object.entries(proto).sort((a, b) => b[1] - a[1]),
  };
}

/** header + string rows -> array of record objects (for streaming replay) */
export function rowsToRecords(header: string[], rows: string[][]): Record<string, string>[] {
  return rows.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}
