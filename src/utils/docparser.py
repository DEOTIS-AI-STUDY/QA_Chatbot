import os
from docx import Document
from docx.oxml import ns

def extract_hyperlinks(paragraph):
    nsmap = ns.nsmap
    links = []
    for elem in paragraph._element.findall('.//w:hyperlink', nsmap):
        rId = elem.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        url = None
        text = ""
        if rId:
            rel = paragraph.part.rels.get(rId)
            if rel:
                url = rel.target_ref
        for t in elem.findall('.//w:t', nsmap):
            text += t.text or ""
        if url:
            links.append((text, url))
    return links

def get_cell_text(cell):
    return "\n".join([p.text for p in cell.paragraphs]).strip()

def parse_table(table):
    lines = []
    rows = table.rows
    if len(rows) < 2:
        return lines
    header = [get_cell_text(cell).replace('\n', ' ') for cell in rows[0].cells]
    md_header = "| " + " | ".join(header) + " |"
    md_sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines.append(md_header)
    lines.append(md_sep)
    for row in rows[1:]:
        row_items = []
        for col_idx, cell in enumerate(row.cells):
            if col_idx < len(header):
                value = get_cell_text(cell).replace('\n', ' ')
                row_items.append(value)
        lines.append("| " + " | ".join(row_items) + " |")
    lines.append("")
    lines.append("")
    return lines

def parse_docx_file(filepath):
    doc = Document(filepath)
    nsmap = ns.nsmap
    result_lines = []
    tables_blocks = []
    img_count = 1

    block_items = []
    for block in doc.element.body:
        if block.tag.endswith('tbl'):
            for table in doc.tables:
                if table._tbl == block:
                    block_items.append(('table', table))
                    break
        elif block.tag.endswith('p'):
            for para in doc.paragraphs:
                if para._p == block:
                    block_items.append(('para', para))
                    break

    skip_table_indices = set()
    # 표와 표 위 2단락 추출
    for idx, (kind, item) in enumerate(block_items):
        if kind == 'table':
            # 표 위 2단락 추출
            prev_paras = []
            for i in range(max(0, idx-2), idx):
                if block_items[i][0] == 'para':
                    prev_paras.append(block_items[i][1].text.strip())
            table = item
            table_lines = parse_table(table)
            # 괄호로 묶어서 저장
            block = "{\n"
            for para_text in prev_paras:
                if para_text:
                    block += para_text + "\n"
            block += "\n".join(table_lines)
            block += "\n}\n"
            tables_blocks.append(block)
            # 표 위 2단락은 data.txt에 남기고, 표는 제거
            skip_table_indices.add(idx)
        elif kind == 'para':
            para = item
            text = para.text.strip()
            if text:
                result_lines.append(text)
            links = extract_hyperlinks(para)
            if links:
                for link_text, url in links:
                    if link_text.strip().lower().startswith("http://") or link_text.strip().lower().startswith("https://"):
                        continue
                    else:
                        result_lines.append(f"{link_text}({url})")
            for drawing in para._element.findall('.//w:drawing', nsmap):
                docpr = drawing.find('.//wp:docPr', {'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'})
                img_name = docpr.attrib.get('name') if docpr is not None else f"그림 {img_count}"
                img_descr = docpr.attrib.get('descr') if docpr is not None else ""
                hlink = drawing.find('.//a:hlinkClick', {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
                url = ""
                if hlink is not None:
                    rId = hlink.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                    if rId:
                        rel = doc.part.rels.get(rId)
                        if rel:
                            url = rel.target_ref
                if url:
                    result_lines.append(f"이미지:{img_name}, desc:{img_descr}, url:{url}")
                else:
                    result_lines.append(f"이미지:{img_name}, desc:{img_descr}")
                img_count += 1

    # 표 바로 위 2단락은 data.txt에 남기고, 표는 제거
    # 이미 result_lines에 표 위 2단락이 들어가 있으므로, 표 내용만 data.txt에서 빠짐

    return "\n".join(result_lines), tables_blocks

def main():
    data_dir = "./data/"
    output_file = "data/data.txt"
    tables_file = "data/tables.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    docx_files = [f for f in files if f.lower().endswith('.docx')]

    all_parsed = []
    all_tables = []
    for filename in docx_files:
        filepath = os.path.join(data_dir, filename)
        parsed, tables_blocks = parse_docx_file(filepath)
        all_parsed.append(f"--- {filename} ---\n{parsed}\n")
        for block in tables_blocks:
            all_tables.append(f"--- {filename} ---\n{block}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_parsed))
    with open(tables_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_tables))
    print(f"모든 docx 결과가 {output_file}에 저장되었습니다.")
    print(f"모든 표와 표 위 2단락 결과가 {tables_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()

class DocParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data_txt = None
        self.tables_txt = None
        self._parse_file()

    def _parse_file(self):
        # 기존 parse_docx_file 함수 활용
        text, tables_blocks = parse_docx_file(self.filepath)
        self.data_txt = text
        self.tables_txt = tables_blocks

    def get_text(self):
        return self.data_txt

    def get_tables(self):
        # tables_txt는 이미 {}로 묶인 블록 리스트임
        result = []
        for block in self.tables_txt:
            # {} 안 내용만 추출
            start = block.find('{')
            end = block.rfind('}')
            if start != -1 and end != -1:
                content = block[start+1:end].strip()
                result.append(content)
        return result