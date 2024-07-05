import codecs
import os
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-Pd9FqzqbITfp7KXpB0YHWngqXK4GWZvB5BSAf9IoiNDBeie4"

from llama_parse import LlamaParse
from llama_parse.base import ResultType, Language

pdf_file_name = "/home/transwarp/Documents/脱敏材料/北互小微/3.证据1-《个人经营贷授信借款合同》.pdf"

parser = LlamaParse(result_type=ResultType.MD,
                    language=Language.SIMPLIFIED_CHINESE,
                    verbose=True,
                    num_workers=1)

documents = parser.load_data(pdf_file_name)

# Check loaded documents

print(f"Number of documents: {len(documents)}")

with codecs.open("./pdf_parse/LlamaParse/个人经营贷授信借款合同.txt", "w", "utf-8") as fw:
    for doc in documents:
        # print(doc.doc_id)
        fw.write(doc.text)

