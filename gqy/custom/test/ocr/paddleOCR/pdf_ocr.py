from pathlib import Path
from paddleocr import PPStructureV3
from pdf2image import convert_from_path

input_file = "data/保理合同(客户付息).pdf"
output_path = Path("./output")

# 明确指定poppler路径（请替换为你的实际bin目录）
poppler_path = r"D:\workPrograms\poppler-24.08.0\Library\bin"

try:
    # 1. PDF每页转图片，设置dpi提高清晰度
    images = convert_from_path(input_file, poppler_path=poppler_path, dpi=340)
except Exception:
    print("请确认poppler已安装，并将bin目录路径正确填写到poppler_path变量。")
    print("下载地址：https://github.com/oschwartz10612/poppler-windows/releases/")
    exit(1)

image_files = []
image_dir = output_path / "pdf_images"
image_dir.mkdir(parents=True, exist_ok=True)
for idx, img in enumerate(images):
    img_path = image_dir / f"page_{idx+1}.png"
    img.save(img_path)
    image_files.append(str(img_path))

# 2. 交给模型识别每页图片
pipeline = PPStructureV3()
markdown_list = []
markdown_images = []

for img_path in image_files:
    output = pipeline.predict(input=img_path)
    for res in output:
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

# 3. 汇总到md文档
markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)