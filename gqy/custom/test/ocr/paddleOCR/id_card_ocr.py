from paddleocr import PaddleOCR
import json
import re

def extract_idcard_info(rec_texts):
    """
    从 OCR 识别文本中提取身份证结构化信息
    """
    info = {
        "姓名": "",
        "性别": "",
        "民族": "",
        "出生": "",
        "住址": "",
        "公民身份号码": ""
    }
    address_lines = []
    for text in rec_texts:
        if text.startswith("姓名"):
            info["姓名"] = text.replace("姓名", "").strip()
        elif text.startswith("性别"):
            info["性别"] = text.replace("性别", "").strip()
        elif text.startswith("民族"):
            info["民族"] = text.replace("民族", "").strip()
        elif text.startswith("出生"):
            # 出生字段有可能单独一行，日期在下一行
            continue
        elif re.match(r"\d{4}年\d{1,2}月\d{1,2}日", text):
            info["出生"] = text.strip()
        elif text.startswith("住址"):
            address_lines.append(text.replace("住址", "").strip())
        elif re.match(r"^\d{17}[\dXx]$", text):
            info["公民身份号码"] = text.strip()
        elif text.startswith("s"):
            continue
        else:
            # 住址可能分多行
            address_lines.append(text.strip())
    if address_lines:
        info["住址"] = "".join(address_lines)
    return info

# 从保存的json文件读取OCR结果
with open("output/2_res.json", "r", encoding="utf-8") as f:
    ocr_json = json.load(f)

rec_texts = ocr_json.get("rec_texts", [])
idcard_info = extract_idcard_info(rec_texts)
print("结构化身份证信息：")
print(json.dumps(idcard_info, ensure_ascii=False, indent=2))
