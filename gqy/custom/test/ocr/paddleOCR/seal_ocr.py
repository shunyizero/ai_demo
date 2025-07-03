from paddleocr import SealRecognition

pipeline = SealRecognition(
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
)
# ocr = SealRecognition(device="gpu") # 通过 device 指定模型推理时使用 GPU
output = pipeline.predict("./保理合同(客户付息).pdf")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/")
    res.save_to_json("./output/")