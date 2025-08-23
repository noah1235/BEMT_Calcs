import pathlib
import re

def format_float(x: float, width: int, prec: int) -> str:
    s = f"{x:+.{prec}f}" if x < 0 else f"{x:.{prec}f}"
    return s.rjust(width)

def adjust_header_re(lines, re_new_abs):
    out = []
    scaled = re_new_abs / 1e6
    style_1 = re.compile(r"(Re\s*=\s*)([-+0-9.eE]+)\s*e\s*6")
    style_2 = re.compile(r"(Re\s*=\s*)([0-9.eE+-]+)")
    for line in lines:
        if "Re" in line:
            if style_1.search(line):
                line = style_1.sub(rf"\1{scaled:9.3f} e 6", line)
            elif style_2.search(line):
                line = style_2.sub(rf"\1{re_new_abs:.0f}", line)
        out.append(line)
    return out


def process_polar(file_path, cl_scale, cd_scale, scale_cdp, new_re):
    """
    Read an XFOIL polar, scale CL/CD (and optionally CDp), update header Re, and
    return (header_lines, column_header_line, adjusted_row_lines).
    """
    import re

    def format_float(x: float, width: int, prec: int) -> str:
        # XFOIL-like right-justified formatting; no '+' on positives
        s = f"{x:+.{prec}f}" if x < 0 else f"{x:.{prec}f}"
        return s.rjust(width)

    def adjust_header_re(lines, re_new_abs):
        out = []
        scaled = re_new_abs / 1e6
        style_1 = re.compile(r"(Re\s*=\s*)([-+0-9.eE]+)\s*e\s*6")
        style_2 = re.compile(r"(Re\s*=\s*)([0-9.eE+-]+)")
        for line in lines:
            if "Re" in line:
                if style_1.search(line):
                    line = style_1.sub(rf"\1{scaled:9.3f} e 6", line)
                elif style_2.search(line):
                    line = style_2.sub(rf"\1{re_new_abs:.0f}", line)
            out.append(line)
        return out

    # ---------- read & split ----------
    text_lines = file_path.read_text().splitlines()

    try:
        hdr_idx = next(i for i, ln in enumerate(text_lines)
                       if ln.strip().lower().startswith("alpha"))
    except StopIteration:
        raise SystemExit(f"{file_path}: could not find 'alpha' column header line.")

    header = text_lines[:hdr_idx]
    colhdr = text_lines[hdr_idx]
    body = text_lines[hdr_idx + 1 :]

    # update Re in header
    header = adjust_header_re(header, new_re)

    # column indices (tolerant if some columns are missing)
    cols = colhdr.split()
    def idx(name: str):
        try:
            return cols.index(name)
        except ValueError:
            return None

    i_alpha = idx("alpha")
    i_cl = idx("CL")
    i_cd = idx("CD")
    i_cdp = idx("CDp")
    i_cm = idx("CM")

    if i_alpha is None or i_cl is None or i_cd is None:
        raise SystemExit(f"{file_path}: columns must include at least alpha, CL, CD.")

    adjusted_rows = []
    for ln in body:
        # keep blank lines as-is
        if not ln.strip():
            adjusted_rows.append(ln)
            continue

        parts = ln.split()

        # --- skip the dashed separator line and any non-numeric 'alpha' rows ---
        tok = parts[i_alpha] if i_alpha < len(parts) else ""
        # allow floats like -8.250 ; reject strings like '------'
        def _is_float(s: str) -> bool:
            try:
                float(s)
                return True
            except ValueError:
                return False
        if not _is_float(tok):
            adjusted_rows.append(ln)   # preserve the dashed line
            continue

        # parse row (robust to missing optional columns)
        try:
            alpha = float(parts[i_alpha])
            cl = float(parts[i_cl])
            cd = float(parts[i_cd])
            cdp = float(parts[i_cdp]) if i_cdp is not None and i_cdp < len(parts) else None
            cm  = float(parts[i_cm])  if i_cm  is not None and i_cm  < len(parts) else None
        except ValueError:
            # if some row is malformed, keep it unchanged
            adjusted_rows.append(ln)
            continue

        # scale
        if cl > 0:
            cl_new  = cl * cl_scale
            cd_new  = cd * cd_scale
        else:
            cl_new = cl
            cd_new = cd
        cdp_new = cdp
        cm_new  = cm  # unchanged

        # write back with XFOIL-like widths
        parts_fmt = parts[:]  # copy so we only overwrite known fields

        def set_fmt(idx_, val, w, p):
            if idx_ is not None and idx_ < len(parts_fmt) and val is not None:
                parts_fmt[idx_] = format_float(val, w, p).strip()

        set_fmt(i_alpha, alpha, 7, 3)
        set_fmt(i_cl,    cl_new, 8, 4)
        set_fmt(i_cd    ,cd_new,  9, 5)
        if i_cdp is not None and cdp_new is not None:
            set_fmt(i_cdp, cdp_new, 9, 5)
        if i_cm is not None and cm_new is not None:
            set_fmt(i_cm, cm_new, 8, 4)

        # keep spacing tidy; add two leading spaces like typical XFOIL dumps
        adjusted_rows.append("  " + " ".join(s.rjust(8) for s in parts_fmt))

    return header, colhdr, adjusted_rows


def main():
    # ======= USER SETTINGS =======
    folder = pathlib.Path("airfoil_data/Eppler E71")
    src_re = 5e4
    src_ncrit = 9
    new_re = 1e4


    cl_scale = 0.6    # 10% less lift
    cd_cl_scale = .5
    cd_scale = cl_scale / cd_cl_scale
    scale_cdp = False  # also scale CDp
    # =============================

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        m = re.match(r"(.+)_polar_Re(\d+)_Ncrit(\d+)", file_path.stem)
        if not m:
            continue
        name, re_val, ncrit_val = m.groups()
        re_val, ncrit_val = int(re_val), int(ncrit_val)
        if re_val != src_re or ncrit_val != src_ncrit:
            continue

        header, colhdr, rows = process_polar(file_path, cl_scale, cd_scale, scale_cdp, new_re)
        new_name = f"{name}_polar_Re{new_re}_Ncrit{ncrit_val}{file_path.suffix}"
        out_path = folder / new_name
        with open(out_path, "w") as f:
            for ln in header:
                f.write(ln + "\n")
            f.write("\n")
            f.write(colhdr + "\n")
            for ln in rows:
                f.write(ln + "\n")
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
